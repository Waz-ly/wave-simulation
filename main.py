# https://www.pixeled.site/projects/wave-sim

import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
import os
import wave
from scipy.io import wavfile
import pygame

# global vars
simulation_fps = 15
delta_t = 1/simulation_fps

audio_sample_rate = 8096
# goes to around 1000 hz
listening_time = 12

impulse_point = (268, 225)

wall_damping_factor = 0.7

WIDTH = 800
HEIGHT = 450
BUFFER = 15

# global funcs
def setup(folder: str) -> None:
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = folder + '/' + file
            newPath = folder + '/audio/' + file[:-4] + '.wav'
            if not file.startswith('.') and not os.path.isfile(newPath):
                ffmpeg.input(path).output(newPath, loglevel='quiet', preset='ultrafast').run(overwrite_output=1)

def convert_to_audio(data: np.ndarray, newSampleRate, originalSampleRate) -> np.ndarray:
    if data.ndim == 2:
        audio = np.add(data[:, 0], data[:, 1])
    else:
        audio = data

    return audio[np.linspace(0, audio.shape[0], newSampleRate*audio.shape[0]//originalSampleRate, endpoint=False, dtype=int)]

def update_space(current_space, past_space, walls, delta_time, wavespeed, damping_factor):
    curve_of_space = np.copy(current_space)
    
    curve_of_space[1:-1, 1:-1] = (    current_space[1:-1,  :-2]
                                  +   current_space[ :-2, 1:-1]
                                  - 4*current_space[1:-1, 1:-1]
                                  +   current_space[2:  , 1:-1]
                                  +   current_space[1:-1, 2:  ] )
    
    future_space = wavespeed**2 * delta_time**2 * curve_of_space - past_space + 2*current_space
    future_space += damping_factor*current_space - damping_factor*future_space

    # wall collisions
    valid_neighbors = 1 - walls
    neighbor_count = sum(np.roll(valid_neighbors, shift, axis=(0, 1)) for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)])
    neighbor_count[neighbor_count == 0] = 1
    neighbor_sum = sum(np.roll(future_space * valid_neighbors, shift, axis=(0, 1)) for shift in [(0, 1), (0, -1), (1, 0), (-1, 0)])

    future_space = np.where(walls, wall_damping_factor * (neighbor_sum / neighbor_count), future_space)

    # border collisions
    future_space[0, :] = wall_damping_factor*future_space[1, :]
    future_space[-1, :] = wall_damping_factor*future_space[-2, :]
    future_space[:, 0] = wall_damping_factor*future_space[:, 1]
    future_space[:, -1] = wall_damping_factor*future_space[:, -2]

    return future_space, current_space

def build_walls(width, height, mode='audio'):
    walls = np.zeros((width, height), dtype=bool)

    if mode == 'audio':

        for i in range(width):
            for j in range(height):
                if (i - 40)**2 / 20**2 + (j - 22.5)**2 / 15**2 >= 1 and (i - 40)**2 / (20+3)**2 + (j - 22.5)**2 / (15+3)**2 < 1:
                    walls[i, j] = True

    elif mode == 'visual':

        for i in range(width):
            for j in range(height):
                if (i - 400)**2 / 200**2 + (j - 225)**2 / 150**2 >= 1 and (i - 400)**2 / (200+3)**2 + (j - 225)**2 / (150+3)**2 < 1:
                    walls[i, j] = True

    return walls

def apply_impulse(space, point, strength, mode='audio'):
    if mode == 'audio':

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                weight = np.exp(-(dx**2 + dy**2) / (2))
                space[point[0]//10 + dx, point[1]//10 + dy] += weight * strength

    elif mode == 'visual':

        for dx in range(-4, 5):
            for dy in range(-4, 5):
                weight = np.exp(-(dx**2 + dy**2) / (10))
                space[point[0] + dx, point[1] + dy] += weight * strength

    return space

class Simulation:
    def __init__(self, window, width, height, buffer, delta_time, mode='audio'):
        self.window = window
        self.width = width
        self.height = height
        self.buffer = buffer
        self.delta_time = delta_time
        self.mode = mode

        if self.mode == 'audio':
            self.damping_factor = 0.05
        elif self.mode == 'visual':
            self.damping_factor = 0.001
        else:
            raise Exception("invalid mode selected")

        self.wavespeed = 0.95 / delta_time / np.sqrt(2)

        self.current_space = np.zeros((width, height))
        self.past_space = np.zeros((width, height))
        self.wall_map = np.zeros((width, height))

        self.walls = build_walls(width, height, self.mode)
        self.wall_map[self.walls] = np.NaN
        
        # self.current_space = apply_impulse(self.current_space, impulse_point, 2)

        self.cmap = plt.get_cmap('viridis').copy()
        self.cmap.set_bad(color='black')

    def draw(self):
        pixels = 255*self.cmap(((self.current_space + self.wall_map) + 0.5))[:,:,:3]

        surf = pygame.surfarray.make_surface(pixels)
        self.window.blit(surf, (self.buffer, self.buffer))

    def update(self):
        self.current_space, self.past_space = update_space(self.current_space, self.past_space, self.walls, self.delta_time, self.wavespeed, self.damping_factor)
    
    def apply_impulse(self, strength):
        self.current_space = apply_impulse(self.current_space, impulse_point, strength, mode=self.mode)

    def get_point(self, point):
        speaker_average = 0
        sum_of_weights = 0

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                weight = np.exp(-(dx**2 + dy**2) / (10))
                sum_of_weights += weight
                speaker_average += weight * self.current_space[point[0] + dx, point[1] + dy]
        
        return speaker_average/sum_of_weights

class WaveSimulation:
    def __init__(self, window, width, height, buffer, audio):
        self.simulation = Simulation(window, width, height, buffer, delta_t, mode='visual')
        self.audio = audio

    def play_simulation(self):
        clock = pygame.time.Clock()
        run = True
        i = 0

        while run:
            clock.tick(simulation_fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            if i%4 == 0 and int(i*audio_sample_rate/simulation_fps) < self.audio.shape[0]:
                self.simulation.apply_impulse(self.audio[int(i*audio_sample_rate/simulation_fps)])
            i += 1

            self.simulation.update()
            self.simulation.draw()

            pygame.display.update()

        pygame.quit()

def play_simulation(audio):
    window = pygame.display.set_mode((WIDTH + 2*BUFFER, HEIGHT + 2*BUFFER))

    wave_simulation = WaveSimulation(window, WIDTH, HEIGHT, BUFFER, audio)
    wave_simulation.play_simulation()

if __name__ == '__main__':
    # https://onlinetonegenerator.com

    setup("assets")

    if os.path.exists('assets/audio/audio.wav'):
        path = 'assets/audio/audio.wav'
        sampleRate, data = wavfile.read(path)
        audio = convert_to_audio(data, audio_sample_rate, sampleRate)

    # breaks somewhere around new sample rate of 2000
    wavfile.write('assets/audio/audioQL.wav', audio_sample_rate, audio)
    
    audio = 5 * audio / np.max(audio)

    plt.plot(audio)
    plt.title("original")
    plt.show()

    plt.plot(np.array_split(np.fft.fft(audio), 2)[0])
    plt.title("original_fft")
    plt.show()

    # dummy_window = pygame.display.set_mode((WIDTH//10 + 2*BUFFER, HEIGHT//10 + 2*BUFFER))
    dummy_window = None
    audio_simulation = Simulation(dummy_window, WIDTH//10, HEIGHT//10, BUFFER, 1/audio_sample_rate, mode='audio')

    # run = True
    # i = 0
    # clock = pygame.time.Clock()
    # while run:
    #     clock.tick(30)
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             run = False
    #             break

    #     if i < audio.shape[0]:
    #         audio_simulation.apply_impulse(audio[i])

    #     audio_simulation.update()
    #     audio_simulation.draw()

    #     pygame.display.update()

    #     i += 1

    # pygame.quit()

    mic = []
    for i in range(listening_time * audio_sample_rate):
        if i < audio.shape[0]:
            audio_simulation.apply_impulse(audio[i])
        audio_simulation.update()
        mic.append(audio_simulation.get_point((532//10, 225//10)))

    mic = mic / np.max(mic)

    plt.plot(mic)
    plt.title("final")
    plt.show()

    plt.plot(np.array_split(np.fft.fft(mic), 2)[0])
    plt.title("final_fft")
    plt.show()
    
    left_channel = mic
    right_channel = mic
    speaker = np.array([left_channel, right_channel]).T
    speaker = (speaker * (2 ** 15 - 1)).astype("<h")
    
    with wave.open("assets/audio/audio_heard.wav", "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(audio_sample_rate)
        f.writeframes(speaker.tobytes())

    play_simulation(audio)
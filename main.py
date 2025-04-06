# https://www.pixeled.site/projects/wave-sim

import numpy as np
import matplotlib.pyplot as plt
import pygame
from time import perf_counter
from file_manager import *

# global vars
simulation_fps = 15
delta_t = 1/simulation_fps

audio_sample_rate = 8000
# goes to around 1000 hz
listening_time = 12

wall_damping_factor = 0.9

WIDTH = 800
HEIGHT = 450
BUFFER = 15

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

class Speaker:
    def __init__(self, position, delay, audio_path, rate, multiplier = 1):
        self.position = position
        self.delay = delay * rate
        self.audio = read_audio(audio_path)
        self.max_time = len(self.audio) + self.delay
        self.multiplier = multiplier

    def get_strength(self, time):
        if time > self.delay and time < self.max_time:
            return self.audio[time - self.delay] * self.multiplier
        else:
            return 0
    
    def get_position(self):
        return self.position

class Simulation:
    def __init__(self, window, width, height, buffer, delta_time, mode='audio'):
        self.window = window
        self.width = width
        self.height = height
        self.buffer = buffer
        self.delta_time = delta_time
        self.mode = mode
        self.speakers = []
        self.time_step = 0

        if self.mode == 'audio':
            self.damping_factor = 0.05
        elif self.mode == 'visual':
            self.damping_factor = 0.001
        else:
            raise Exception("invalid mode selected")
        
        if self.mode == 'audio':
            self.wavespeed = 0.75 / self.delta_time / np.sqrt(2)
        elif self.mode == 'visual':
            self.wavespeed = 0.98 / self.delta_time / np.sqrt(2)     

        self.current_space = np.zeros((width, height))
        self.past_space = np.zeros((width, height))
        self.alpha = self.wavespeed**2 * self.delta_time**2 - self.damping_factor
        self.beta = self.damping_factor - 1
        self.gamma = 2 - self.damping_factor

        # wall setup
        self.wall_map = np.zeros((width, height))
        self.walls = build_walls(width, height, self.mode)
        self.not_walls = 1 - self.walls
        self.neighbor_not_wall_count = np.zeros((width, height))
        self.neighbor_not_wall_count[1:-1, 1:-1] = ( self.not_walls[1:-1,  :-2]
                                                   + self.not_walls[ :-2, 1:-1]
                                                   + self.not_walls[2:  , 1:-1]
                                                   + self.not_walls[1:-1, 2:  ] )
        self.neighbor_not_wall_count[self.neighbor_not_wall_count == 0] = 1
        self.neighbor_sum = np.zeros((width, height))
        self.walls_info = (self.walls, self.not_walls, self.neighbor_not_wall_count, self.neighbor_sum)
        self.wall_map[self.walls] = np.NaN
        
        # self.current_space = apply_impulse(self.current_space, impulse_point, 2)

        self.cmap = plt.get_cmap('viridis').copy()
        self.cmap.set_bad(color='black')

    def add_speaker(self, speaker):
        self.speakers.append(speaker)

    def draw(self):
        pixels = 255*self.cmap(((self.current_space + self.wall_map) + 2)/4)[:,:,:3]

        surf = pygame.surfarray.make_surface(pixels)
        self.window.blit(surf, (self.buffer, self.buffer))

    def update(self):
        curve_of_space = np.zeros(self.current_space.shape)
        center_space = self.current_space[1:-1, 1:-1]
    
        curve_of_space[1:-1, 1:-1] = (  self.current_space[1:-1,  :-2]
                                    +   self.current_space[ :-2, 1:-1]
                                    - 4*center_space
                                    +   self.current_space[2:  , 1:-1]
                                    +   self.current_space[1:-1, 2:  ] )

        self.current_space = (self.alpha*curve_of_space
                              + self.beta*self.past_space
                              + self.gamma*self.current_space)
        self.past_space[1:-1, 1:-1] = center_space

        # border collisions
        self.current_space[[0, -1], :] = wall_damping_factor * self.current_space[[1, -2], :]
        self.current_space[:, [0, -1]] = wall_damping_factor * self.current_space[:, [1, -2]]

        # wall collisions
        self.neighbor_sum[1:-1, 1:-1] = ( self.not_walls[1:-1,  :-2] * self.current_space[1:-1,  :-2]
                                    + self.not_walls[ :-2, 1:-1] * self.current_space[ :-2, 1:-1]
                                    + self.not_walls[2:  , 1:-1] * self.current_space[2:  , 1:-1]
                                    + self.not_walls[1:-1, 2:  ] * self.current_space[1:-1, 2:  ] )

        self.current_space = np.where(self.walls, wall_damping_factor * (self.neighbor_sum / self.neighbor_not_wall_count), self.current_space)

        self.update_speakers()

        if self.mode == 'audio':
            self.time_step += 1
        else:
            self.time_step += audio_sample_rate//simulation_fps
    
    def apply_impulse(self, point, strength):
        if self.mode == 'audio':
            
            self.current_space[point[0]//10, point[1]//10] += strength

        elif self.mode == 'visual':

            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    weight = np.exp(-(dx**2 + dy**2) / (10))
                    self.current_space[point[0] + dx, point[1] + dy] += weight * strength

    def update_speakers(self):
        for speaker in self.speakers:
            self.apply_impulse(speaker.get_position(), speaker.get_strength(self.time_step))

    def get_point(self, point):
        return self.current_space[point]

class WaveSimulation:
    def __init__(self, window, width, height, buffer, delta_time, speakers, mode):
        self.simulation = Simulation(window, width, height, buffer, delta_time, mode)
        for speaker in speakers:
            self.simulation.add_speaker(speaker)

    def play_simulation(self):
        clock = pygame.time.Clock()
        run = True

        while run:
            clock.tick(simulation_fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            self.simulation.update()
            self.simulation.draw()

            pygame.display.update()

        pygame.quit()

    def extract_audio(self):
        t1 = perf_counter()

        mic = []
        
        for _ in range(listening_time * audio_sample_rate):
            self.simulation.update()
            mic.append(self.simulation.get_point((532//10, 225//10)))

        t2 = perf_counter()

        print("audio simulation completed in:", t2 - t1, "seconds")

        mic = normalize(mic)

        plt.plot(mic)
        plt.title("final")
        plt.show()

        plt.plot(np.abs(np.array_split(np.fft.fft(mic), 2)[0]))
        plt.title("final_fft")
        plt.show()

        write_audio("assets/audio/audio_heard.wav", audio_sample_rate, mic)

def play_simulation(speakers):
    window = pygame.display.set_mode((WIDTH + 2*BUFFER, HEIGHT + 2*BUFFER))

    wave_simulation = WaveSimulation(window, WIDTH, HEIGHT, BUFFER, delta_t, speakers, mode='visual')
    wave_simulation.play_simulation()

def get_audio(speakers):
    audio_simulation = WaveSimulation(None, WIDTH//10, HEIGHT//10, None, 1/audio_sample_rate, speakers, mode='audio')

    audio_simulation.extract_audio()

def play_fullspeed(speakers):
    window = pygame.display.set_mode((WIDTH//10 + 2*BUFFER, HEIGHT//10 + 2*BUFFER))
    
    wave_simulation = WaveSimulation(window, WIDTH//10, HEIGHT//10, BUFFER, 1/audio_sample_rate, speakers, mode='audio')
    wave_simulation.play_simulation()

if __name__ == '__main__':
    # https://onlinetonegenerator.com

    setup("assets", audio_sample_rate)

    speakers = [
        Speaker((268, 225), 0, "assets/audio/audio.wav", audio_sample_rate, 5),
        Speaker((300, 315), 0, "assets/audio/audio.wav", audio_sample_rate),
        Speaker((390, 290), 2, "assets/audio/audio.wav", audio_sample_rate),
        Speaker((510, 300), 5, "assets/audio/audio.wav", audio_sample_rate),
        Speaker((320, 160), 0, "assets/audio/audio.wav", audio_sample_rate),
        Speaker((405, 140), 3, "assets/audio/audio.wav", audio_sample_rate),
        Speaker((504, 145), 0, "assets/audio/audio.wav", audio_sample_rate)
    ]

    play_simulation(speakers)
    # play_fullspeed(speakers)
    # get_audio(speakers)
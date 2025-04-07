import numpy as np
import matplotlib.pyplot as plt
import pygame
from time import perf_counter
from file_manager import *
from configs import *
from Simulation import Simulation

class WaveSimulation:
    def __init__(self, window, speakers, walls, mode):
        self.simulation = Simulation(window, walls, mode)
        for speaker in speakers:
            self.simulation.add_speaker(speaker)

    def play_simulation(self):
        clock = pygame.time.Clock()
        run = True

        while run:
            clock.tick(VIDEO_RATE)
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
        
        for _ in range(LISTENING_TIME * AUDIO_RATE):
            self.simulation.update()
            mic.append(self.simulation.get_point((532 // SCALING, 225 // SCALING)))

        t2 = perf_counter()

        print("audio simulation completed in:", t2 - t1, "seconds")

        mic = normalize(mic)

        plt.plot(mic)
        plt.title("final")
        plt.show()

        plt.plot(np.abs(np.array_split(np.fft.fft(mic), 2)[0]))
        plt.title("final_fft")
        plt.show()

        write_audio("assets/audio/audio_heard.wav", AUDIO_RATE, mic)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    def get_3D_plot(self, time):
        for i in range(time):
            self.simulation.update()
            
        Z = self.simulation.get_space()
        Z = Z[150:-150:5,20:-20:5]
        c = 0.005
        Z = np.power(2, -np.power((Z-c)/c,2))-np.power(2, -np.power((Z+c)/c,2))/100
        X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(X, Y, Z, cmap='Wistia', edgecolor=None)
        ax.plot_wireframe(X, Y, Z, color='orange', linewidth=0.2)

        ax.set_axis_off()
        ax.view_init(elev=30, azim=145)

        plt.savefig("3d_plot.png", transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
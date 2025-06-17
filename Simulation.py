import numpy as np
import matplotlib.pyplot as plt
import pygame
from file_manager import *
from configs import *

class Simulation:
    def __init__(self, window, walls, mode='audio'):
        self.window = window
        self.mode = mode
        self.speakers = []
        self.time = 0

        if self.mode == 'audio':

            self.width = WIDTH // SCALING
            self.height = HEIGHT // SCALING
            self.delta_time = 1/AUDIO_RATE
            self.wavespeed = 0.75 / self.delta_time / np.sqrt(2)

            self.damping = REAL_DAMPING * 343 / 100 * self.delta_time

            self.time_increment = 1

        elif self.mode == 'visual':

            self.width = WIDTH
            self.height = HEIGHT
            self.delta_time = 1/VIDEO_RATE
            self.wavespeed = 0.98 / self.delta_time / np.sqrt(2)

            self.damping = REAL_DAMPING * 343 / 100 * self.delta_time

            self.time_increment = AUDIO_RATE // VIDEO_RATE

        else:

            raise Exception("invalid mode selected")          

        self.current_space = np.zeros((self.width, self.height))
        self.past_space = np.zeros((self.width, self.height))

        self.alpha = self.wavespeed**2 * self.delta_time**2 * np.power(10, -self.damping/10)
        self.beta = - np.power(10, -self.damping/10)
        self.gamma = 1 + np.power(10, -self.damping/10)

        # wall setup
        self.wall_map = np.zeros((self.width, self.height))
        self.walls = walls
        self.not_walls = 1 - self.walls
        self.neighbor_not_wall_count = np.zeros((self.width, self.height))
        self.neighbor_not_wall_count[1:-1, 1:-1] = ( self.not_walls[1:-1,  :-2]
                                                   + self.not_walls[ :-2, 1:-1]
                                                   + self.not_walls[2:  , 1:-1]
                                                   + self.not_walls[1:-1, 2:  ] )
        self.neighbor_not_wall_count[self.neighbor_not_wall_count == 0] = 1
        self.neighbor_sum = np.zeros((self.width, self.height))
        self.walls_info = (self.walls, self.not_walls, self.neighbor_not_wall_count, self.neighbor_sum)
        self.wall_map[self.walls] = np.NaN
        
        # self.current_space = apply_impulse(self.current_space, impulse_point, 2)

        self.cmap = plt.get_cmap('viridis').copy()
        self.cmap.set_bad(color='black')

    def add_speaker(self, speaker):
        self.speakers.append(speaker)

    def draw(self):
        pixels = 255*self.cmap(((self.current_space + self.wall_map) + 3)/6)[:,:,:3]

        surf = pygame.surfarray.make_surface(pixels)
        self.window.blit(surf, (BUFFER, BUFFER))

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
        self.current_space[[0, -1], :] = WALL_DAMPING * self.current_space[[1, -2], :] + (1 - WALL_DAMPING) * self.past_space[[1, -2], :]
        self.current_space[:, [0, -1]] = WALL_DAMPING * self.current_space[:, [1, -2]] + (1 - WALL_DAMPING) * self.past_space[:, [1, -2]]

        # wall collisions
        self.neighbor_sum[1:-1, 1:-1] = ( self.not_walls[1:-1,  :-2] * self.current_space[1:-1,  :-2]
                                    + self.not_walls[ :-2, 1:-1] * self.current_space[ :-2, 1:-1]
                                    + self.not_walls[2:  , 1:-1] * self.current_space[2:  , 1:-1]
                                    + self.not_walls[1:-1, 2:  ] * self.current_space[1:-1, 2:  ] )

        self.current_space = np.where(self.walls, WALL_DAMPING * (self.neighbor_sum / self.neighbor_not_wall_count) + (1 - WALL_DAMPING) * self.past_space, self.current_space)

        self.update_speakers()

        self.time += self.time_increment
    
    def apply_impulse(self, point, strength):
        if self.mode == 'audio':
            
            self.current_space[point[0] // SCALING, point[1] // SCALING] += 16 * strength

        elif self.mode == 'visual':

            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    weight = np.exp(-(dx**2 + dy**2) / (10))
                    self.current_space[point[0] + dx, point[1] + dy] += weight * strength

    def update_speakers(self):
        for speaker in self.speakers:
            self.apply_impulse(speaker.get_position(), speaker.get_strength(self.time))

    def get_point(self, point):
        return self.current_space[point]
    
    def get_space(self):
        return self.current_space
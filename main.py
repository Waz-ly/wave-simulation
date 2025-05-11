# https://www.pixeled.site/projects/wave-sim

import numpy as np
from file_manager import *
from Speaker import Speaker
from configs import *
from player import *

def build_walls(mode):
    if mode == 'visual':

        walls = np.zeros((WIDTH, HEIGHT), dtype=bool)

        for i in range(WIDTH):
            for j in range(HEIGHT):
                if (i - 400)**2 / 200**2 + (j - 225)**2 / 150**2 >= 1 and (i - 400)**2 / (200+3)**2 + (j - 225)**2 / (150+3)**2 < 1:
                    walls[i, j] = True

    if mode == 'audio':

        walls = np.zeros((WIDTH//SCALING, HEIGHT//SCALING), dtype=bool)

        for i in range(WIDTH//SCALING):
            for j in range(HEIGHT//SCALING):
                if (i - 400/SCALING)**2 / (200/SCALING)**2 + (j - 225/SCALING)**2 / (150/SCALING)**2 >= 1 and (i - 400/SCALING)**2 / (200/SCALING+3)**2 + (j - 225/SCALING)**2 / (150/SCALING+3)**2 < 1:
                    walls[i, j] = True
    
    return walls

if __name__ == '__main__':
    # https://onlinetonegenerator.com

    setup("assets", AUDIO_RATE)

    speakers = [
        Speaker((268, 225), 0, "main"),
        Speaker((300, 315), 0.2, "Noise 1"),
        Speaker((390, 290), 1.2, "Noise 2"),
        Speaker((510, 300), 0.7, "Noise 3"),
        Speaker((320, 160), 0.4, "Noise 4"),
        Speaker((405, 140), 1.5, "Noise 5"),
        Speaker((504, 145), 1.4, "Noise 6")
    ]

    walls = build_walls("visual")

    play_simulation(speakers, walls)
    # play_fullspeed(speakers, walls)
    # get_audio(speakers, walls)
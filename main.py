# https://www.pixeled.site/projects/wave-sim

import numpy as np
from file_manager import *
from Speaker import Speaker
from configs import *
from player import *

def build_walls(mode):
    walls = np.zeros((WIDTH, HEIGHT), dtype=bool)

    for i in range(WIDTH):
        for j in range(WIDTH):
            if (i - 400)**2 / 200**2 + (j - 225)**2 / 150**2 >= 1 and (i - 400)**2 / (200+30)**2 + (j - 225)**2 / (150+30)**2 < 1:
                walls[i, j] = True
    
    if mode == 'audio':
        return walls[::10, ::10]
    
    return walls

if __name__ == '__main__':
    # https://onlinetonegenerator.com

    setup("assets", AUDIO_RATE)

    speakers = [
        Speaker((268, 225), 0, "assets/audio/audio.wav", 5),
        Speaker((300, 315), 0, "assets/audio/audio.wav"),
        Speaker((390, 290), 2, "assets/audio/audio.wav"),
        Speaker((510, 300), 5, "assets/audio/audio.wav"),
        Speaker((320, 160), 0, "assets/audio/audio.wav"),
        Speaker((405, 140), 3, "assets/audio/audio.wav"),
        Speaker((504, 145), 0, "assets/audio/audio.wav")
    ]

    walls = build_walls("audio")

    # play_simulation(speakers, walls)
    play_fullspeed(speakers, walls)
    # get_audio(speakers, walls)
from file_manager import *
from configs import *

class Speaker:
    def __init__(self, position, delay, audio_path, multiplier = 1):
        self.position = position
        self.delay = int(delay * AUDIO_RATE)
        self.audio = read_audio("assets/audio/" + audio_path + ".wav")
        self.max_time = len(self.audio) + self.delay
        self.multiplier = multiplier

    def get_strength(self, time):
        if time > self.delay and time < self.max_time:
            return self.audio[time - self.delay] * self.multiplier
        else:
            return 0
    
    def get_position(self):
        return self.position
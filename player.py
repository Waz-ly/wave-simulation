import pygame
from configs import *
from WaveSimulation import WaveSimulation


def play_simulation(speakers, walls):
    window = pygame.display.set_mode((WIDTH + 2*BUFFER, HEIGHT + 2*BUFFER))

    wave_simulation = WaveSimulation(window, speakers, walls, mode='visual')
    wave_simulation.play_simulation()

def get_audio(speakers, walls):
    audio_simulation = WaveSimulation(None, speakers, walls, mode='audio')

    audio_simulation.extract_audio()

def play_fullspeed(speakers, walls):
    window = pygame.display.set_mode((WIDTH // SCALING + 2*BUFFER, HEIGHT//10 + 2*BUFFER))
    
    wave_simulation = WaveSimulation(window, speakers, walls, mode='audio')
    wave_simulation.play_simulation()
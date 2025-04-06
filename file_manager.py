import numpy as np
import ffmpeg
import os
from scipy.io import wavfile

def setup(folder: str, audio_sample_rate) -> None:
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = folder + '/' + file
            newPath = folder + '/audio/' + file[:-4] + '.wav'
            if not file.startswith('.') and not os.path.isfile(newPath):
                ffmpeg.input(path).output(newPath, loglevel='quiet', preset='ultrafast').run(overwrite_output=1)

            sampleRate, data = read_audio(newPath, includeSampleRate=True)
            audio = downsample(data, audio_sample_rate, sampleRate)
            audio = normalize(audio)
            write_audio(newPath, audio_sample_rate, audio)

def read_audio(path, includeSampleRate=False):
    data = wavfile.read(path)[1]

    if data.ndim == 2:
        audio = np.add(data[:, 0], data[:, 1])
    else:
        audio = data

    if includeSampleRate:
        return wavfile.read(path)[0], audio
    else:
        return audio
    
def normalize(audio):
    return audio / np.max(audio)

def write_audio(path, sample_rate, audio):
    wavfile.write(path, sample_rate, audio)

def downsample(data: np.ndarray, newSampleRate, originalSampleRate) -> np.ndarray:
    return data[np.linspace(0, data.shape[0], newSampleRate*data.shape[0]//originalSampleRate, endpoint=False, dtype=int)]
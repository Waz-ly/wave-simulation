import numpy as np
import ffmpeg
import os
import wave

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
    with wave.open(path, 'rb') as f:

        sample_rate = f.getframerate()
        channels = f.getnchannels()
        data = f.readframes(f.getnframes())

        audio = np.frombuffer(data, dtype="<h")
        audio = audio.reshape(-1, channels) / (2 ** 15 - 1)

    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if includeSampleRate:
        return sample_rate, audio

    return audio
    
def normalize(audio):
    return audio / np.max(np.abs(audio))

def write_audio(path, sample_rate, audio):
    audio = np.array([audio, audio]).T
    audio = (audio * (2 ** 15 - 1)).astype("<h")

    with wave.open(path, "w") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio.tobytes())

def downsample(data: np.ndarray, newSampleRate, originalSampleRate) -> np.ndarray:
    return data[np.linspace(0, data.shape[0], newSampleRate*data.shape[0]//originalSampleRate, endpoint=False, dtype=int)]
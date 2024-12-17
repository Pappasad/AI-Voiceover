import preprocessing as pp
import pygame
import librosa
import os
import numpy as np
import soundfile as sf
import sys
from pydub import AudioSegment
import matplotlib.pyplot as plt

def showWave(path):
    y, sr = librosa.load(path, sr=None)

    plt.figure(figsize=(10,4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


def playNWait(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): pass

def test1(path='testclipped.wav'):
    print("Running Test 1...")

    #playNWait(path)
    showWave(path)

    audio = AudioSegment.from_wav(path)

    audio, sample_rate = librosa.load(path, sr=None)

    audio = pp.eliminateNoise(audio)

    sf.write('testdenoised.wav', audio, sample_rate)

    print("Playing denoised...")

    showWave('testdenoised.wav')

def test2():
    pp.preprocessMain()


if __name__ == '__main__':
    test2()

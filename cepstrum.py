import wave

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, messagebox

if __name__ == "__main__":
    # Select a wav file
    file = filedialog.askopenfilename(initialdir="/DWOO/python_speech_recognition-main/data/wav", title="Select file",
                                      filetypes=(("wav files", "*.wav"), )
                                      )
    if file == "":
        messagebox.showerror("Error", "No file selected")
        exit()

    target_time = 0.58

    fft_size = 1024
    cep_threshold = 33
    with wave.open(file) as wav:
        sampling_freq = wav.getframerate()  # Get Hz
        waveform = wav.readframes(wav.getnframes())  # Get waveform
        waveform = np.frombuffer(waveform, dtype=np.int16)  # Convert to int16\

    target_idx = np.int(target_time * sampling_freq)
    frame = waveform[target_idx:target_idx + fft_size].copy()

    frame = frame * np.hamming(fft_size)
    spectrum = np.fft.fft(frame, fft_size)
    log_power = 2 * np.log(np.abs(spectrum[:int(fft_size / 2) + 1]) + 1e-7)
    cepstrum = np.fft.ifft(log_power)
    cepstrum_low = cepstrum.copy()
    cepstrum_low[cep_threshold+1:-(cep_threshold)] = 0.0
    log_power_ceplo = np.abs(np.fft.fft(cepstrum_low))
    cepstrum_high = cepstrum - cepstrum_low
    cepstrum_high[0] = cepstrum[0]

    low_power_cephi = np.abs(np.fft.fft(cepstrum_high))
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

    # Open the wav file
    wav = wave.open(file, "r")
    frame_size = 25
    frame_shift = 10
    sampling_freq = wav.getframerate()  # Get Hz
    sample_size = wav.getsampwidth()  # Get byte
    num_channels = wav.getnchannels()  # Get number of channels
    num_samples = wav.getnframes()  # Get number of samples
    waveform = wav.readframes(num_samples)  # Get waveform
    waveform = np.frombuffer(waveform, dtype=np.int16)  # Convert to int16

    frame_size = int(sampling_freq * frame_size / 1000) # Convert ms to sample
    frame_shift = int(sampling_freq * frame_shift / 1000) # Convert ms to sample

    fft_size = 1
    while fft_size < frame_size:
        fft_size *= 2

    num_frames = (num_samples - frame_size) // frame_shift + 1
    print("Number of frames: {}".format(num_frames))

    spectrogram = np.zeros((num_frames, int(fft_size / 2) + 1))

    for frame_idx in range(num_frames):
        start_idx = frame_idx * frame_shift
        frame = waveform[start_idx:start_idx + frame_size].copy()
        frame = frame * np.hamming(frame_size)
        spectrum = np.fft.fft(frame, fft_size)
        absolute = np.abs(spectrum)
        absolute = absolute[:int(fft_size / 2) + 1]
        log_absolute = np.log(absolute + 1e-7)
        spectrogram[frame_idx, :] = log_absolute

    fig = plt.figure(figsize=(10, 10))  # Create figure
    plt.subplot(2, 1, 1)
    # show waveform on first subplot
    time_axis = np.arange(num_samples) / sampling_freq  # Create time axis
    plt.plot(time_axis, waveform)  # Plot waveform
    plt.xlabel("Time [s]")  # Label of x axis
    plt.ylabel("Amplitude")  # Label of y axis
    plt.xlim([0, num_samples/sampling_freq])  # Set range of x axis

    # show spectrogram on second subplot
    plt.subplot(2, 1, 2)
    spectrogram -= np.max(spectrogram)
    vmax = np.abs(np.min(spectrogram)) * 0.0
    vmin = -np.abs(np.min(spectrogram)) * 0.7
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "winter"
    colormapping = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.imshow(spectrogram.T[-1::-1, :],
               extent=[0, num_samples/sampling_freq, 0, sampling_freq/2],
                aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Spectrogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    cbar = fig.colorbar(colormapping, ax=plt.gca())

    plt.show()

    wav.close()


    # spectrogram 그대로 출력
    plt.subplots(1, 5, figsize=(50, 10))
    freq_axis = np.arange(fft_size / 2 + 1) / fft_size * sampling_freq
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.plot(freq_axis, spectrogram[i])
        plt.title("Spectrogram")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
    plt.show()
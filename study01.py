import wave
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, messagebox

if __name__ == "__main__":
    # Select a wav file
    file = filedialog.askopenfilename(initialdir="/", title="Select file",
                                      filetypes=(("wav files", "*.wav"), )
                                      )
    if file == "":
        messagebox.showerror("Error", "No file selected")
        exit()

    # Open the wav file
    wav = wave.open(file, "r")
    sampling_freq = wav.getframerate()  # Get Hz
    sample_size = wav.getsampwidth()  # Get byte
    num_channels = wav.getnchannels()  # Get number of channels
    num_samples = wav.getnframes()  # Get number of samples
    waveform = wav.readframes(num_samples)  # Get waveform
    waveform = np.frombuffer(waveform, dtype=np.int16)  # Convert to int16

    # print information
    print("Sampling frequency: {} Hz".format(sampling_freq))
    print("Sample size: {} byte".format(sample_size))
    print("Number of channels: {}".format(num_channels))
    print("Number of samples: {}".format(num_samples))

    time_axis = np.arange(num_samples) / sampling_freq  # Create time axis
    print("Time axis: {} s".format(time_axis))
    plt.figure(figsize=(10, 4))  # Create figure
    plt.plot(time_axis, waveform)  # Plot waveform
    plt.xlabel("Time [s]")  # Label of x axis
    plt.ylabel("Amplitude")  # Label of y axis

    plt.xlim([0, num_samples/sampling_freq])  # Set range of x axis
    plt.show()

    # Close the wav file
    wav.close()

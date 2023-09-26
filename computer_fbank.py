import wave

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, messagebox


class FeatureExtractor():
    def __init__(self, sample_frequency=16000,
                 frame_length=25, frame_shift=10,
                 num_mel_bins=23, num_ceps=13,
                 lifter_coef=22, low_frequency=20,
                 high_frequency=8000, dither=1.0):
        self.sample_freq = sample_frequency
        self.frame_size = int(sample_frequency * frame_length / 1000)
        self.frame_shift = int(sample_frequency * frame_shift / 1000)
        self.num_mel_bins = num_mel_bins
        self.num_ceps = num_ceps
        self.lifter_coef = lifter_coef
        self.low_freq = low_frequency
        self.high_freq = high_frequency
        self.dither_coef = dither

        self.fft_size = 1
        while self.fft_size < self.frame_size:
            self.fft_size *= 2

        self.mel_filterbank = self._create_mel_filterbank()

    def _herz_to_mel(self, herz):
        return 1127.0 * np.log(1.0 + herz / 700.0)

    def _create_mel_filterbank(self):
        mel_high_freq = self._herz_to_mel(self.high_freq)
        mel_low_freq = self._herz_to_mel(self.low_freq)
        mel_points = np.linspace(mel_low_freq,
                                 mel_high_freq,
                                 self.num_mel_bins + 2)
        dim_spectrum = int(self.fft_size / 2) + 1
        mel_filterbank = np.zeros((self.num_mel_bins, dim_spectrum))
        for mel_index in range(self.num_mel_bins):
            left_mel = mel_points[mel_index]
            center_mel = mel_points[mel_index + 1]
            right_mel = mel_points[mel_index + 2]
            for freq_index in range(dim_spectrum):
                freq = freq_index * (self.sample_freq / 2) / dim_spectrum
                mel = self._herz_to_mel(freq)
                if mel > left_mel and mel < right_mel:
                    if mel <= center_mel:
                        weight = (mel - left_mel) / (center_mel - left_mel)
                    else:
                        weight = (right_mel - mel) / (right_mel - center_mel)
                    mel_filterbank[mel_index, freq_index] = weight
        return mel_filterbank

    def _extract_window(self, waveform, start_idx, num_samples):
        window = waveform[start_idx:start_idx + self.frame_size].copy()

        # Dithering
        if self.dither_coef > 0:
            window = window \
                     + 2 * self.dither_coef * np.random.rand(len(window)) \
                     - self.dither_coef

        # DC offset elimination
        window = window - np.mean(window)
        power = np.sum(window**2)
        if power < 1E-10:
            power = 1E-10
        log_power = np.log(power)

        # Pre-emphasis
        # window[i] = 1.0 * window[i] - 0.97 * window[i-1]
        window = np.convolve(window, np.array([1.0, -0.97]), mode="same")
        window[0] -= 0.97 * window[0]

        # Hamming window
        window = window * np.hamming(self.frame_size)

        return window, log_power


    def _compute_fbank(self, waveform):
        num_samples = np.size(waveform)
        num_frames = (num_samples - self.frame_size) // self.frame_shift + 1
        fbank_features = np.zeros((num_frames, self.num_mel_bins))
        log_power = np.zeros(num_frames)

        # Compute features frame by frame
        for frame in range(num_frames):
            start_idx = frame * self.frame_shift
            window, log_power[frame] = self._extract_window(waveform,
                                                            start_idx,
                                                            num_samples)
            spectrum = np.fft.fft(window, self.fft_size)
            spectrum = np.abs(spectrum[:int(self.fft_size / 2) + 1]) ** 2
            # (1, fft_size/2+1).dot((fft_size/2+1, num_mel_bins)) = (1, num_mel_bins)
            fbank = np.dot(spectrum, self.mel_filterbank.T)
            fbank[fbank < 0.1] = 0.1 # Floor for log
            fbank_features[frame] = np.log(fbank)
        return fbank_features, log_power


if __name__ == "__main__":

    # Select a wav file
    file = filedialog.askopenfilename(initialdir="/DWOO/python_speech_recognition-main/data/wav", title="Select file",
                                      filetypes=(("wav files", "*.wav"),)
                                      )
    if file == "":
        messagebox.showerror("Error", "No file selected")
        exit()


    sample_freq = 16000
    frame_length = 25
    frame_shift = 10
    low_freq = 20
    high_freq = sample_freq / 2
    num_mel_bins = 10
    dither = 1.0

    feature_extractor = FeatureExtractor(sample_freq,
                                          frame_length,
                                          frame_shift,
                                          num_mel_bins,
                                         13,
                                          low_freq,
                                          high_freq,
                                          8000,
                                          dither)

    # Open the wav file
    wav = wave.open(file, "r")
    num_samples = wav.getnframes()  # Get number of samples
    waveform = wav.readframes(num_samples)  # Get waveform
    waveform = np.frombuffer(waveform, dtype=np.int16)  # Convert to int16

    fbank_features, log_power = feature_extractor._compute_fbank(waveform)
    (num_frames, num_mel_bins) = fbank_features.shape

    fbank_features = fbank_features.astype(np.float32)

    print("Number of frames: {}".format(num_frames))
    print("Number of mel bins: {}".format(num_mel_bins))
    print("Log power: {}".format(log_power))
    print("FBank features: {}".format(fbank_features))

    wav.close()
import wave

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
from tkinter import filedialog, messagebox

def dp_matching(feature1, feature2):
    (n_frames1, num_dims) = np.shape(feature1)
    n_frames2 = np.shape(feature2)[0]

    distance = np.zeros((n_frames1, n_frames2))
    for n in range(n_frames1):
        for m in range(n_frames2):
            distance[n, m] = np.sum((feature1[n] - feature2[m]) ** 2)

    cost = np.zeros((n_frames1, n_frames2))
    track = np.zeros((n_frames1, n_frames2)) # 0: down, 1: diagonal, 2: right
    cost[0, 0] = distance[0, 0]
    for n in range(1, n_frames1):
        cost[n, 0] = cost[n-1, 0] + distance[n, 0]
        track[n, 0] = 0
    for m in range(1, n_frames2):
        cost[0, m] = cost[0, m-1] + distance[0, m]
        track[0, m] = 2

    for n in range(1, n_frames1):
        for m in range(1, n_frames2):
            vertical = cost[n-1, m] + distance[n, m]
            diagonal = cost[n-1, m-1] + 2 * distance[n, m]
            horizontal = cost[n, m-1] + distance[n, m]

            candidate = [vertical, diagonal, horizontal]
            transition = np.argmin(candidate)

            cost[n, m] = candidate[transition]
            track[n, m] = transition

    total_cost = cost[-1, -1] / (n_frames1 + n_frames2)

    path = []
    n = n_frames1 - 1
    m = n_frames2 - 1
    while True:
        path.append((n, m))
        if n == 0 and m == 0:
            break

        if track[n, m] == 0:
            n -= 1
        elif track[n, m] == 1:
            n -= 1
            m -= 1
        elif track[n, m] == 2:
            m -= 1

    path.reverse()

    return total_cost, path

if __name__ == "__main__":
    # Select a wav file
    file1 = filedialog.askopenfilename(initialdir="/", title="Select file",
                                      filetypes=(("bin file", "*.bin"), )
                                      )
    file2 = filedialog.askopenfilename(initialdir="/", title="Select file",
                                       filetypes=(("bin file", "*.bin"),)
                                       )
    if file1 == "" or file2 == "":
        messagebox.showerror("Error", "No file selected")
        exit()

    result = './alignment.txt'

    num_dims = 13
    mfcc_1 = np.fromfile(file1, dtype=np.float32)
    mfcc_2 = np.fromfile(file2, dtype=np.float32)

    total_cost, min_path = dp_matching(mfcc_1, mfcc_2)

    with open(result, 'w') as f:
        for p in min_path:
            f.write('{} {}\n'.format(p[0], p[1]))

import os

import numpy as np

from hmmfunc import MonoPhoneHMM

if __name__ == "__main__":
    hmmproto = './exp/model_3state_1mix/hmmproto'
    mean_std_file = '../01compute_features/mfcc/train_small/mean_std.txt'
    out_dir = os.path.dirname(hmmproto)

    with open(mean_std_file, 'r') as f:
        lines = f.readlines()
        mean_line = lines[1]
        std_line = lines[3]

        mean = mean_line.split()
        std = std_line.split()

        mean = np.array(mean, dtype=np.float64)
        std = np.array(std, dtype=np.float64)

        var = std ** 2

    hmm = MonoPhoneHMM()
    hmm.load_hmm(hmmproto)
    hmm.flat_init(mean, var)
    hmm.save_hmm(os.path.join(out_dir, '0.hmm'))
    
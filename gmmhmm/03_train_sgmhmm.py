import os
import numpy as np

from hmmfunc import MonoPhoneHMM

if __name__ == "__main__":
    base_hmm = './exp/model_3state_1mix/0.hmm'
    feat_scp = '../01compute_features/mfcc/train_small/feats.scp'
    label_file = './exp/data/train_small/text_int'

    num_iter = 10
    num_utters = 50

    out_dir = os.path.dirname(base_hmm)

    hmm = MonoPhoneHMM()
    hmm.load_hmm(base_hmm)

    label_list = {}
    with open(label_file, 'r') as f:
        for line in f:
            utt = line.split()[0]
            lab = line.split()[1:]
            lab = np.int64(lab)
            label_list[utt] = lab

    feat_list = {}
    with open(feat_scp, 'r') as f:
        for n, line in enumerate(f):
            if n >= num_utters:
                break
            utt = line.split()[0]
            ff = line.split()[1]
            feat_list[utt] = ff

    for iter in range(num_iter):
        hmm.train(feat_list, label_list)
        out_hmm = os.path.join(out_dir, "%d.hmm" % (iter + 1))
        hmm.save_hmm(out_hmm)

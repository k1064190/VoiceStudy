import os

from hmmfunc import MonoPhoneHMM

if __name__ == "__main__":
    phone_list_file = './exp/data/train_small/phone_list'
    num_states = 3
    num_dims = 13
    prob_loop = 0.7
    out_dir = './exp/model_%dstate_1mix' % (num_states)

    phone_list = []
    with open(phone_list_file, mode='r') as f:
        for line in f:
            phone = line.split()[0]
            phone_list.append(phone)

    hmm = MonoPhoneHMM()
    hmm.make_proto(phone_list, num_states, prob_loop, num_dims)
    hmm.save_hmm(os.path.join(out_dir, 'hmmproto'))

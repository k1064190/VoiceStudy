import json
import os

import numpy as np


class MonoPhoneHMM():
    def __init__(self):
        self.phones = []
        self.num_phones = 1
        self.num_states = 1
        self.num_mixture = 1
        self.num_dims = 1
        self.pdf = None
        self.trans = None
        self.LZERO = -1E10
        self.LSMALL = -0.5E10
        self.ZERO = 1E-100
        self.MINVAL = 1E-4

        self.elem_prob = None
        self.state_prob = None
        self.alpha = None
        self.beta = None
        self.loglikelihood = 0
        self.pdf_accumulators = None
        self.trans_accumulators = None

        # viterbi
        self.score = None
        self.track = None
        self.viterbi_score = 0

    def make_proto(self, phone_list, num_states,
                   prob_loop, num_dims):
        self.phones = phone_list
        self.num_phones = len(phone_list)
        self.num_states = num_states
        self.num_mixture = 1

        # pdf[o][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_s = []
                for m in range(self.num_mixture):
                    mu = np.zeros(self.num_dims)
                    var = np.ones(self.num_dims)
                    weight = 1.0
                    gconst = self.calc_gconst(var)
                    gaussian = {
                        'weight' : weight,
                        'mu' : mu,
                        'var' : var,
                        'gConst' : gconst
                    }
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            self.pdf.append(tmp_p)

        # trans[p][s] = [loop, next]
        prob_next = 1.0 - prob_loop
        log_prob_loop = np.log(prob_loop) \
                        if prob_loop > self.ZERO else self.LZERO
        log_prob_next = np.log(prob_next) \
                        if prob_next > self.ZERO else self.LZERO

        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            for s in range(self.num_states):
                tmp_trans = np.array([log_prob_loop, log_prob_next])
                tmp_p.append(tmp_trans)
            self.trans.append(tmp_p)


    def calc_gconst(self, var):
        return self.num_dims * np.log(2 * np.pi) \
                + np.sum(np.log(var))

    def flat_init(self, mean, var):
        for p in range(self.num_phones):
            for s in range(self.num_states):
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    pdf['mu'] = mean
                    pdf['var'] = var
                    pdf['gconst'] = self.calc_gconst(var)

    def save_hmm(self, filename):
        ''' HMMパラメータをjson形式で保存
        filename: 保存ファイル名
        '''
        # json形式で保存するため，
        # HMMの情報を辞書形式に変換する
        hmmjson = {}
        # 基本情報を入力
        hmmjson['num_phones'] = self.num_phones
        hmmjson['num_states'] = self.num_states
        hmmjson['num_mixture'] = self.num_mixture
        hmmjson['num_dims'] = self.num_dims
        # 音素モデルリスト
        hmmjson['hmms'] = []
        for p, phone in enumerate(self.phones):
            model_p = {}
            # 音素名
            model_p['phone'] = phone
            # HMMリスト
            model_p['hmm'] = []
            for s in range(self.num_states):
                model_s = {}
                # 状態番号
                model_s['state'] = s
                # 遷移確率(対数値から戻す)
                model_s['trans'] = \
                    list(np.exp(self.trans[p][s]))
                # GMMリスト
                model_s['gmm'] = []
                for m in range(self.num_mixture):
                    model_m = {}
                    # 混合要素番号
                    model_m['mixture'] = m
                    # 混合重み
                    model_m['weight'] = \
                        self.pdf[p][s][m]['weight']
                    # 平均値ベクトル
                    # jsonはndarrayを扱えないので
                    # list型に変換しておく
                    model_m['mean'] = \
                        list(self.pdf[p][s][m]['mu'])
                    # 対角共分散
                    model_m['variance'] = \
                        list(self.pdf[p][s][m]['var'])
                    # gConst
                    model_m['gConst'] = \
                        self.pdf[p][s][m]['gConst']
                    # gmmリストに加える
                    model_s['gmm'].append(model_m)
                # hmmリストに加える
                model_p['hmm'].append(model_s)
            # 音素モデルリストに加える
            hmmjson['hmms'].append(model_p)

        # JSON形式で保存する
        with open(filename, mode='w') as f:
            json.dump(hmmjson, f, indent=4)

    def load_hmm(self, filename):
        ''' json形式のHMMファイルを読み込む
                filename: 読み込みファイル名
                '''
        # JSON形式のHMMファイルを読み込む
        with open(filename, mode='r') as f:
            hmmjson = json.load(f)

        # 辞書の値を読み込んでいく
        self.num_phones = hmmjson['num_phones']
        self.num_states = hmmjson['num_states']
        self.num_mixture = hmmjson['num_mixture']
        self.num_dims = hmmjson['num_dims']

        # 音素情報の読み込み
        self.phones = []
        for p in range(self.num_phones):
            hmms = hmmjson['hmms'][p]
            self.phones.append(hmms['phone'])

        # 遷移確率の読み込み
        # 音素番号p, 状態番号s の遷移確率は
        # trans[p][s] = [loop, next]
        self.trans = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                hmm = hmms['hmm'][s]
                # 遷移確率の読み込み
                tmp_trans = np.array(hmm['trans'])
                # 総和が1になるよう正規化
                tmp_trans /= np.sum(tmp_trans)
                # 対数に変換
                for i in [0, 1]:
                    tmp_trans[i] = np.log(tmp_trans[i]) \
                        if tmp_trans[i] > self.ZERO \
                        else self.LZERO
                tmp_p.append(tmp_trans)
            # self.transに追加
            self.trans.append(tmp_p)

        # 正規分布パラメータの読み込み
        # 音素番号p, 状態番号s, 混合要素番号m
        # の正規分布はpdf[p][s][m]でアクセスする
        # pdf[p][s][m] = gaussian
        self.pdf = []
        for p in range(self.num_phones):
            tmp_p = []
            hmms = hmmjson['hmms'][p]
            for s in range(self.num_states):
                tmp_s = []
                hmm = hmms['hmm'][s]
                for m in range(self.num_mixture):
                    gmm = hmm['gmm'][m]
                    # 重み，平均，分散，gConstを取得
                    weight = gmm['weight']
                    mu = np.array(gmm['mean'])
                    var = np.array(gmm['variance'])
                    gconst = gmm['gConst']
                    # 正規分布を作成
                    gaussian = {'weight': weight,
                                'mu': mu,
                                'var': var,
                                'gConst': gconst}
                    tmp_s.append(gaussian)
                tmp_p.append(tmp_s)
            # self.pdfに追加
            self.pdf.append(tmp_p)


    def calc_pdf(self, pdf, obs):
        tmp = (obs - pdf['mu']) **2 / pdf['var']
        if np.ndim(tmp) == 2:
            # obs = (frame x dim)
            tmp = np.sum(tmp, axis=1)
        elif np.ndim(tmp) == 1:
            tmp = np.sum(tmp)
        logprob = -0.5 * (tmp + pdf['gConst'])
        return logprob

    def logadd(self, x, y):
        # x = log(a)
        # y = log(b)
        # z = log(a+b)
        if x > y:
            z = x + np.log(1.0 + np.exp(y - x))
        else:
            z = y + np.log(1.0 + np.exp(x - y))
        return z

    def calc_out_prob(self, feat, label):
        # feat : 1발화 분량의 특징값
        # label : 1발화 분량의 레이블
        feat_len = np.shape(feat)[0]
        label_len = len(label)

        # elem_prob[l][s][m][t]
        self.elem_prob = np.zeros((label_len,
                                   self.num_states,
                                   self.num_mixture,
                                   feat_len))
        self.state_prob = np.zeros((label_len,
                                    self.num_states,
                                    feat_len))

        for l, p in enumerate(label):
            for s in range(self.num_states):
                self.state_prob[l][s][:] = \
                self.LZERO * np.ones(feat_len)
                for m in range(self.num_mixture):
                    pdf = self.pdf[p][s][m]
                    self.elem_prob[l][s][m][:] = \
                        self.calc_pdf(pdf, feat)
                    tmp_prob = np.log(pdf['weight']) \
                        + self.elem_prob[l][s][m][:]
                    for t in range(feat_len):
                        self.state_prob[l][s][t] = \
                        self.logadd(self.state_prob[l][s][t],
                                    tmp_prob[t])

    def calc_alpha(self, label):
        



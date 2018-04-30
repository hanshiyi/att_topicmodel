import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import model.w2vloader as w2vld
import model.negtopic as negtopic
import model.utils as postopic
import random
import time
import numpy as np
from numpy.random import choice
from sklearn.metrics import accuracy_score
from model.logger import Logger

def main():
    iter = 0
    MaxIteration = 50000
    topic_num = 50
    neg_num = 5
    pos_num = 0
    epsilon = 0.2
    mode = 2
    topic_dim = 30
    winsiz = 2
    using_gpu = 2
    batch_size = 5
    learning_rate = 0.0025
    cuda_valid = torch.cuda.is_available()
    # cuda_valid = False
    w2v_model = w2vld.w2vloader('data/top1000020newsglove', cuda_valid, using_gpu)
    # file2 = open('data/testVocabList_freq.txt', 'r')
    file2 = open('/home/shiyi/nvdm/data/20news/vocab.new', 'r')
    wordlist = [line.strip().split()[0] for line in file2.readlines() if line.strip().split()[0] in w2v_model.w2idx]
    # linelist = w2v_model.normalize(linelist)
    pospart = negtopic.MUSE_sum(w2v_model.dim, topic_num, 10041).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(
        w2v_model.dim, topic_num, winsiz)
    pospart.load_state_dict(
        torch.load('/home/shiyi/rlfortopicmodel/model/20newsmuse814/test/pospartmuse3389'))
    pospart.w2idx = w2v_model.w2idx
    linelist = pospart.load_wordembed(wordlist)
    topic_pr = pospart.predict(linelist)
    outfile = open('data/wordmatrix814.txt','w')
    for idx, line in enumerate(wordlist):
        outfile.write(line)
        for i in topic_pr[idx]:
            outfile.write(' '+str(i.data.cpu().numpy().tolist()[0]))
        outfile.write('\n')
    print topic_pr

if __name__ == '__main__':
    main()
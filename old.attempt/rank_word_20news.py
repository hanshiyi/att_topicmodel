import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import utils.w2vloader as w2vld
import utils.negtopic as negtopic
import utils.leftModule as postopic
import random
import time
import heapq
import numpy as np


def main():
    iter = 0
    MaxIteration = 100
    topic_num = 50
    neg_num = 10
    pos_num = 0
    epsilon = 0.8
    topic_dim = 300
    emb_dim = 300
    winsiz = 3
    using_gpu = 3
    batch_size = 5
    cuda_valid = torch.cuda.is_available()
    # cuda_valid = False
    w2v_model = w2vld.w2vloader('data/top1000020newsglove', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    # wordfile = open('data/wholeVocabList+freq_top10000_813.txt', 'r')
    wordfile = open('/home/shiyi/nvdm/data/20news/vocab.new', 'r')
    # topicfile = open('data/20news-bydate-train-processed/metadata.tsv', 'r')
    # topicfile.readline()
    # topiclist = []
    # for to in topicfile.readlines():
    #     topiclist.append(to.strip().split()[1])
    pospart = negtopic.MUSE_sum(emb_dim, topic_num, 10041).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(
        emb_dim, topic_num, 10041)
    pospart.w2idx = w2v_model.w2idx
    pospart.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newsmuse818/test/pospartmuse201'))
    dis_nparray = []
    def to_np(x):
        return x.data.cpu().numpy()

    wordlist = [line.strip().split()[0] for line in wordfile.readlines() if line.strip().split()[0] in pospart.w2idx]
    linelist = pospart.load_wordembed(wordlist)
    topic_pr = pospart.predict(linelist).data.cpu().numpy()
    dis_nparray = np.transpose(topic_pr,(1,0))
    for idx, row in enumerate(dis_nparray):
        top_word = heapq.nlargest(10, range(len(row)), row.take)
        print str(idx) + ':',
        for word_id in top_word:
            print wordlist[word_id],
        print ''




if __name__ == '__main__':
    main()
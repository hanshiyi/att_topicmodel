import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import utils.w2vloader as w2vld
import utils.negtopic as negtopic
import utils.leftModule as postopic
import random
import time
import numpy as np


def main():
    iter = 0
    MaxIteration = 100
    topic_num = 25
    neg_num = 10
    pos_num = 0
    epsilon = 0.8
    topic_dim = 30
    winsiz = 3
    using_gpu = 3
    batch_size = 5
    # cuda_valid = torch.cuda.is_available()
    cuda_valid = False
    w2v_model = w2vld.w2vloader('data/glove.42B.300d.txt', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = 'data/20news-bydate-test-processed'
    pospart = negtopic.negtopic(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.negtopic(w2v_model.dim, topic_num, winsiz)
    pospart.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newsmuse84/pospartmuse49999'))
    topicfile = open('data/20news-bydate-train-processed/metadata.tsv', 'r')
    topicfile.readline()
    topiclist = []
    for to in topicfile.readlines():
        topiclist.append(to.strip().split()[1])
    wordfile = open('data/vocabList', 'r')
    wordlist = []
    for line in wordfile.readlines():
        wordlist.append(line.strip())

    def move_forward(model, data):
        h_last = Variable(torch.zeros(2, batch_size, w2v_model.dim/2)).cuda(using_gpu) if cuda_valid else Variable(torch.zeros(2, batch_size, w2v_model.dim/2))
        cen_idx = torch.autograd.Variable(torch.LongTensor([winsiz])).cuda(using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([winsiz]))
        topic_distribution = model(data, h_last, cen_idx)
        return topic_distribution

    def to_np(x):
        return x.data.cpu().numpy()

    try:
        avg_acc = 0.0
        for iter in range(MaxIteration):
            cur_win, nxt_win, y_topic = postopic.BatchPick(batch_size, corpus_path, winsiz_list)
            pos_context_emb = w2v_model.load_wordembed(cur_win)
            pos_topic_dis, pos_sf_dis = move_forward(pospart, pos_context_emb)
            pred_num, pred_topic = torch.max(pos_topic_dis,1)
            accuracy = (torch.squeeze(torch.FloatTensor(y_topic)) == torch.squeeze(
                torch.FloatTensor(to_np(pred_topic))))

            for idx, right in enumerate(accuracy):
                if right:
                    print topiclist[pred_topic[idx].data[0]],
                    print cur_win[idx]

        print str(iter) + 'iterations\' accuracy: ',
        print avg_acc/MaxIteration

    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'

if __name__ == '__main__':
    main()
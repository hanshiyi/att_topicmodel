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
    topic_num = 20
    winsiz = 5
    using_gpu = 3
    batch_size = 5
    # cuda_valid = torch.cuda.is_available()
    cuda_valid = False
    w2v_model = w2vld.w2vloader('data/glove.42B.300d.txt', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = 'data/20news-bydate-test-processed'
    pospart = negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz)
    pospart.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newsnocatbigwindow/july30/pospartgru41640'))
    print 'loading completed'
    def move_forward(model, data):
        h_last = Variable(torch.zeros(2, batch_size, w2v_model.dim / 2)).cuda(using_gpu) if cuda_valid else Variable(
            torch.zeros(2, batch_size, w2v_model.dim / 2))
        cen_idx = torch.autograd.Variable(torch.LongTensor([winsiz])).cuda(
            using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([winsiz]))
        topic_distribution = model(data, h_last, cen_idx)
        return topic_distribution

    def to_np(x):
        return x.data.cpu().numpy()

    try:
        cur_win, y_topic = postopic.TestPickAll(corpus_path)
        avg_acc = 0.0
        for idx, document in enumerate(cur_win):
            # winsiz_list = [winsiz for _ in xrange(batch_size)]
            batch_size = 1
            # cur_win, nxt_win, y_topic = postopic.BatchPick(batch_size, corpus_path, winsiz_list)
            pos_context_emb = w2v_model.load_wordembed(document)
            pos_topic_dis = move_forward(pospart, torch.unsqueeze(pos_context_emb,0))
            pred_topic = torch.max(pos_topic_dis,1)[1]
            accuracy = (torch.squeeze(torch.FloatTensor([y_topic[idx]])) == torch.squeeze(
                torch.FloatTensor(to_np(pred_topic)))).float().mean()
            avg_acc += accuracy

        print 'iterations\' accuracy: ',
        print avg_acc/len(cur_win)

    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'

if __name__ == '__main__':
    main()
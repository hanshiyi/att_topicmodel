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
from numpy.random import choice
from sklearn.metrics import accuracy_score
from utils.logger import Logger

def main():
    iter = 0
    MaxIteration = 100000
    topic_num = 50
    neg_num = 5
    pos_num = 0
    bag_num = 2
    epsilon = 0.25
    mode = 2
    topic_dim = 300
    winsiz = 2
    using_gpu = 2
    batch_size = 6
    learning_rate = 0.0025
    cuda_valid = torch.cuda.is_available()
    # cuda_valid = False
    w2v_model = w2vld.w2vloader('data/lilittleglovetraintest', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = 'data/20news-bydate-train-processed-glovewords'
    topic_represention_U = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    topic_represention_V = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    negpart = negtopic.MUSE_sum(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(w2v_model.dim, topic_num, winsiz)
    pospart = negtopic.MUSE_sum(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(w2v_model.dim, topic_num, winsiz)
    optimizer_pos = optim.Adam(pospart.parameters(), lr=learning_rate)
    # pospart.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newsmuse84/augest6sigwithnormandep/pospartmuse49999'))
    jjtest = topic_represention_U.weight.data
    pospart.fn.weight.data.copy_(jjtest)
    negpart.copy_from(pospart)
    optimizer_U = optim.Adam(topic_represention_U.parameters(), lr=learning_rate)
    optimizer_V = optim.Adam(topic_represention_V.parameters(), lr=learning_rate)
    logger = Logger('./logger/20newsmuse88miba')

    def move_forward(model, data):
        h_last = Variable(torch.zeros(2, batch_size, w2v_model.dim/2)).cuda(using_gpu) if cuda_valid else Variable(torch.zeros(2, batch_size, w2v_model.dim/2))
        cen_idx = torch.autograd.Variable(torch.LongTensor([winsiz])).cuda(using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([winsiz]))
        topic_distribution, st_distribution = model(data, h_last, cen_idx)
        return topic_distribution, st_distribution

    def LogLossFun(t_pos, t_nxt, t_neg):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = [topic_represention_V(bag) for bag in t_nxt]
        logloss = []
        for idx, posV in enumerate(posVec):
            nxtV = [bagVec[idx] for bagVec in nxtVec]#posV: 3*30   nxtV:3*30
            negV = negVec[idx]#negV: 10*30
            # inter1 = torch.sum(nxtV.mul(posV))
            # inter2 = torch.sigmoid(inter1)
            add1 = 0
            for bagnxt in nxtV:
                add1 += -torch.log(torch.sigmoid(torch.sum(bagnxt.mul(posV))) + 1e-13)
            inter3 = torch.sum(-negV.mm(posV.transpose(0, 1)),0)
            add2 = torch.sum(-torch.log(torch.sigmoid(inter3)+1e-13))
            logloss.append(add1 + add2)
        return logloss  #batchsize

    def CrsEtpLossFun(t_pos, t_nxt, t_neg, topic_distribution):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = [topic_represention_V(bag) for bag in t_nxt]
        crsetploss = [[] for _ in xrange(batch_size)]
        for idx, posV in enumerate(posVec):
            q_value = torch.index_select(topic_distribution[idx],0,t_pos[idx])  #topic_distribution[idx]: 1*25(topic_num), t_pos[idx]: 1
            q_value_neg = torch.index_select(topic_distribution[idx],0,t_neg[idx])
            nxtV = nxtVec[idx]  # posV: 1*30   nxtV:1*30
            negV = negVec[idx]  # negV: 5(neg_num)*30
            Lcap_nxt = torch.sigmoid(nxtV.mm(posV.transpose(0, 1)))
            Lcap_neg = torch.sigmoid(negV.mm(posV.transpose(0, 1)))
            crsetploss_idx = 0
            for i, lc in enumerate(Lcap_nxt):
                crsetploss_idx = -lc * torch.log(q_value + 1e-13) - (1-lc) * torch.log(1 - q_value + 1e-13)
            for i, ln in enumerate(Lcap_neg):
                crsetploss_idx += -ln * torch.log(q_value_neg[i] + 1e-13) - (1-ln) * torch.log(1 - q_value_neg[i] + 1e-13)
            crsetploss[idx].append(crsetploss_idx)
        return crsetploss   #batchsize*1

    def to_np(x):
        return x.data.cpu().numpy()

    def sample_topic(topic_distribution):
        if mode == 1:
            ans_lst = []
            for pos_dis_dis in topic_distribution:
                ans_lst.append(list(choice(topic_num, 1, p=to_np(pos_dis_dis))))
            ans_lst = Variable(torch.LongTensor(ans_lst)).cuda(using_gpu) if cuda_valid else Variable(
                torch.LongTensor(ans_lst))
        elif mode == 2:
            if random.random() > epsilon:
                ans_lst = torch.max(topic_distribution, 1)[1]
            else:
                ans_lst = [[np.random.randint(topic_num)] for _ in range(batch_size)]
                ans_lst = Variable(torch.LongTensor(ans_lst)).cuda(using_gpu) if cuda_valid else Variable(
                    torch.LongTensor(ans_lst))
        return ans_lst
    def sample_neg(pos_ed, nxt_ed):
        tep_lst = []
        for idx, pos in enumerate(pos_ed):
            ori_lst = set([i for i in range(topic_num)])
            tep_lst.append(random.sample(ori_lst-set(pos_ed[idx])-set(nxt_ed[idx]), neg_num))
        return tep_lst

    try:
        avg_accuracy = 0.
        for iter in range(MaxIteration):
            winsiz_list = [winsiz for _ in xrange(batch_size)]
            cur_win, nxt_win, y_topic = postopic.BatchMBPick(batch_size, corpus_path, winsiz_list, bag_num)
            pos_context_emb = w2v_model.load_wordembed(cur_win)
            nxt_context_emb = w2v_model.load_wordembed(nxt_win)
            pos_topic_dis, pos_sf_dis = move_forward(pospart, pos_context_emb)
            bags_topic_dis, bags_sf_dis = [],[]
            for bag in nxt_context_emb:
                nxt_topic_dis, nxt_sf_dis = move_forward(negpart, bag)
                bags_topic_dis.append(nxt_topic_dis)
                bags_sf_dis.append(nxt_sf_dis)

            pos_temp_list = []
            t_pos = sample_topic(pos_sf_dis)
            bag_temp_list = []
            for sf_dis in bags_sf_dis:
                bag_temp_list.append(sample_topic(sf_dis))
            t_nxt = bag_temp_list
            t_neg = Variable(torch.LongTensor(sample_neg(t_pos, t_nxt))).cuda(using_gpu) if cuda_valid else Variable(torch.LongTensor(sample_neg(t_pos, t_nxt)))
            logLoss = LogLossFun(t_pos, t_nxt, t_neg)
            optimizer_U.zero_grad()
            optimizer_V.zero_grad()
            loss1 = 0.
            for item in logLoss:
                item.backward(retain_variables=True)
                loss1 += item
            loss2 = 0.
            optimizer_U.step()
            optimizer_V.step()
            crsetcrstepVal = CrsEtpLossFun(t_pos, t_nxt, t_neg, pos_topic_dis)
            optimizer_pos.zero_grad()
            for crete in crsetcrstepVal:
                for item in crete:
                    for i in item:
                        loss2 += i
                        i.backward(retain_variables=True)
            if iter % 100 == 0:
                print loss2
                info = {
                    'loss1': loss1.data[0],
                    'loss2': loss2.data[0]
                }
                avg_accuracy = 0.
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, iter + 1)

                for tag, value in pospart.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), iter + 1)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), iter + 1)

                negpart.copy_from(pospart)

            optimizer_pos.step()
    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'
        torch.save(pospart.state_dict(), 'model/20newsmuse88/pospartminibag' + str(iter))
        torch.save(topic_represention_U, 'model/20newsmuse88/rUminibag'+str(iter))
        torch.save(topic_represention_V, 'model/20newsmuse88/rVminibag'+str(iter))
        exit()
    torch.save(pospart.state_dict(), 'model/20newsmuse88/pospartminibag' + str(iter))
    torch.save(topic_represention_U, 'model/20newsmuse88/rUminibag'+str(iter))
    torch.save(topic_represention_V, 'model/20newsmuse88/rVminibag'+str(iter))

if __name__ == '__main__':
    main()
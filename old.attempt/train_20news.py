import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import model.w2vloader as w2vld
import model.negtopic as negtopic
import model.leftModule as postopic
import random
import time
import numpy as np
from sklearn.metrics import accuracy_score
from model.logger import Logger

def main():
    iter = 0
    MaxIteration = 120000
    topic_num = 25
    neg_num = 5
    pos_num = 0
    epsilon = 0.8
    topic_dim = 30
    winsiz = 5
    using_gpu = 3
    batch_size = 5
    learning_rate = 0.0025
    cuda_valid = torch.cuda.is_available()
    # cuda_valid = False
    w2v_model = w2vld.w2vloader('data/glove.42B.300d.txt', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = 'data/20news-bydate-train-processed'
    topic_represention_U = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    topic_represention_V = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    negpart = negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz)
    pospart = negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz).cuda(using_gpu) if cuda_valid else negtopic.negtopic_nocat(w2v_model.dim, topic_num, winsiz)
    optimizer_pos = optim.Adam(pospart.parameters(), lr=learning_rate)
    negpart.copy_from(pospart)
    optimizer_U = optim.Adam(topic_represention_U.parameters(), lr=learning_rate)
    optimizer_V = optim.Adam(topic_represention_V.parameters(), lr=learning_rate)
    logger = Logger('./logger/20newsnocat1')

    def move_forward(model, data):
        h_last = Variable(torch.zeros(2, batch_size, w2v_model.dim/2)).cuda(using_gpu) if cuda_valid else Variable(torch.zeros(2, batch_size, w2v_model.dim/2))
        cen_idx = torch.autograd.Variable(torch.LongTensor([winsiz])).cuda(using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([winsiz]))
        topic_distribution = model(data, h_last, cen_idx)
        return topic_distribution

    def LogLossFun(t_pos, t_nxt, t_neg):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = topic_represention_V(t_nxt)
        logloss = []
        for idx, posV in enumerate(posVec):
            nxtV = nxtVec[idx]#posV: 3*30   nxtV:3*30
            negV = negVec[idx]#negV: 10*30
            inter1 = torch.sum(nxtV.mul(posV))
            inter2 = torch.sigmoid(inter1)
            add1 = -torch.log(inter2 + 1e-13)
            inter3 = torch.sum(-negV.mm(posV.transpose(0, 1)),0)
            add2 = torch.sum(-torch.log(torch.sigmoid(inter3)+1e-13))
            logloss.append(add1 + add2)
        return logloss  #batchsize

    def CrsEtpLossFun(t_pos, t_nxt, t_neg, topic_distribution):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = topic_represention_V(t_nxt)
        crsetploss = [[] for _ in xrange(batch_size)]
        for idx, posV in enumerate(posVec):
            q_value = torch.index_select(topic_distribution[idx],0,t_pos[idx])
            nxtV = nxtVec[idx]  # posV: 3*30   nxtV:3*30
            negV = negVec[idx]  # negV: 10*30
            Lcap = torch.sigmoid(nxtV.mm(posV.transpose(0, 1)))
            for i, lc in enumerate(Lcap):
                crsetploss_idx = -lc * torch.log(q_value + 1e-13) - (1-lc) * torch.log(1 - q_value + 1e-13)
                crsetploss[idx].append(crsetploss_idx)
        return crsetploss   #batchsize*3

    def CrsEtpLossFun_attemptone(t_pos, t_nxt, t_neg, topic_distribution):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = topic_represention_V(t_nxt)
        temp = torch.autograd.Variable(torch.LongTensor([i for i in xrange(topic_num)])).cuda(using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([i for i in xrange(topic_num)]))
        embv = topic_represention_V(temp)
        crsetploss = [[] for _ in xrange(batch_size)]
        softmax = nn.Softmax()
        for idx, posV in enumerate(posVec):
            dis = topic_distribution[idx]
            q_value = torch.index_select(topic_distribution[idx],0,t_pos[idx])
            dis = nn.functional.softmax(torch.div(dis, q_value.expand(topic_num)))
            nxtV = nxtVec[idx]  # posV: 1*30   nxtV:1*30
            Lcap = torch.sigmoid(embv.mm(posV.transpose(0, 1)))
            for i, lc in enumerate(Lcap):
                if i == t_pos[idx].data[0]:
                    continue
                crsetploss_idx = -lc * torch.log(dis[i] + 1e-13) - (1-lc) * torch.log(1 - dis[i] + 1e-13)
                crsetploss[idx].append(crsetploss_idx)
        return crsetploss   #batchsize*3

    def to_np(x):
        return x.data.cpu().numpy()

    def sample_topic(topic_distribution):
        ans_lst = []
        random.seed(time.time())
        while True:
            if len(ans_lst) <= pos_num:
                bolzman_number = random.random()
                for idx, num in enumerate(topic_distribution):
                    if bolzman_number < num.data[0]:
                        if idx not in ans_lst:
                            ans_lst.append(idx)
                        break
                    else:
                        bolzman_number -= num.data[0]
            else:
                break
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
            cur_win, nxt_win, y_topic = postopic.BatchPick(batch_size, corpus_path, winsiz_list)
            pos_context_emb = w2v_model.load_wordembed(cur_win)
            nxt_context_emb = w2v_model.load_wordembed(nxt_win)
            pos_topic_dis = move_forward(pospart, pos_context_emb)
            nxt_topic_dis = move_forward(negpart, nxt_context_emb)
            pos_temp_list = []
            pred_topic = torch.max(pos_topic_dis,1)[1]
            for pos_dis in pos_topic_dis:
                pos_temp_list.append(sample_topic(pos_dis))
            t_pos = Variable(torch.LongTensor(y_topic)).cuda(using_gpu) if cuda_valid else Variable(
                torch.LongTensor(y_topic))
            nxt_temp_list = []
            for nxt_dis in nxt_topic_dis:
                nxt_temp_list.append(sample_topic(nxt_dis))
            t_nxt = Variable(torch.LongTensor(nxt_temp_list)).cuda(using_gpu) if cuda_valid else Variable(
                torch.LongTensor(nxt_temp_list))
            t_neg = Variable(torch.LongTensor(sample_neg(pos_temp_list, nxt_temp_list))).cuda(using_gpu) if cuda_valid else Variable(torch.LongTensor(sample_neg(pos_temp_list, nxt_temp_list)))
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
            # accuracy = (torch.squeeze(torch.FloatTensor(y_topic)) == torch.squeeze(
            #     torch.FloatTensor(to_np(pred_topic)))).float().mean()
            # avg_accuracy += accuracy
            if iter % 100 == 0:
                print loss2
                # avg_accuracy = avg_accuracy/100.0
                # print avg_accuracy
                info = {
                    'loss1': loss1.data[0],
                    'loss2': loss2.data[0]
                    # 'accuracy': avg_accuracy
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
        torch.save(pospart.state_dict(), 'model/20newsnocatbigwindow/pospartgru' + str(iter))
        torch.save(topic_represention_U.state_dict(), 'model/20newsnocatbigwindow/top_repU' + str(iter))
        torch.save(topic_represention_U.state_dict(), 'model/20newsnocatbigwindow/top_repV' + str(iter))
    torch.save(pospart.state_dict(), 'model/20newsnocatbigwindow/pospartgru' + str(iter))
    torch.save(topic_represention_U.state_dict(), 'model/20newsnocatbigwindow/top_repU' + str(iter))
    torch.save(topic_represention_U.state_dict(), 'model/20newsnocatbigwindow/top_repV' + str(iter))

if __name__ == '__main__':
    main()
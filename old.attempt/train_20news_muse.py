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
    MaxIteration = 1000
    topic_num = 50
    neg_num = 48
    pos_num = 0
    word_num = 10041
    emb_dim = 300
    epsilon = 0.8
    mode = 2
    topic_dim = 300
    winsiz = 3
    using_gpu = 3
    batch_size = 64
    learning_rate = 0.0025
    cuda_valid = torch.cuda.is_available()
    # cuda_valid = False
    # w2v_model = w2vld.w2vloader('data/lilittleglovetraintest', cuda_valid, using_gpu)
    # w2v_model = w2vld.w2vloader('data/w2v20news.txt', cuda_valid, using_gpu)
    corpus_path = "data/20news-bydate-train2000"
    topic_represention_U = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    topic_represention_V = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num, topic_dim)
    negpart = negtopic.MUSE_sum(emb_dim, topic_num, word_num).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(emb_dim, topic_num, word_num)
    pospart = negtopic.MUSE_sum(emb_dim, topic_num, word_num).cuda(using_gpu) if cuda_valid else negtopic.MUSE_sum(emb_dim, topic_num, word_num)
    pospart.dataloader('data/top1000020newsglove')
    pospart.load_state_dict(torch.load('model/20newsmuse818/pospartmuse79'))
    topic_represention_U = torch.load('model/20newsmuse818/rU79')
    topic_represention_V = torch.load('model/20newsmuse818/rV79')
    optimizer_pos = optim.Adam(pospart.parameters(), lr=learning_rate)
    pospart.cuda_valid = cuda_valid
    negtopic.cuda_valid = cuda_valid
    # pospart.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newsmuse84/augest6sigwithnormandep/pospartmuse49999'))
    jjtest = topic_represention_U.weight.data
    pospart.fn.data.copy_(jjtest)
    negpart.copy_from(pospart)
    optimizer_U = optim.Adam(topic_represention_U.parameters(), lr=learning_rate)
    optimizer_V = optim.Adam(topic_represention_V.parameters(), lr=learning_rate)
    logger = Logger('./logger/20newsmuse818')

    def move_forward(model, data):
        h_last = Variable(torch.zeros(2, batch_size, emb_dim/2)).cuda(using_gpu) if cuda_valid else Variable(torch.zeros(2, batch_size, emb_dim.dim/2))
        cen_idx = torch.autograd.Variable(torch.LongTensor([winsiz])).cuda(using_gpu) if cuda_valid else torch.autograd.Variable(torch.LongTensor([winsiz]))
        topic_distribution, st_distribution = model(data, h_last, cen_idx)
        return topic_distribution, st_distribution

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
            q_value = torch.index_select(topic_distribution[idx],0,t_pos[idx])  #topic_distribution[idx]: 1*25(topic_num), t_pos[idx]: 1
            q_value_neg = torch.index_select(topic_distribution[idx],0,t_neg[idx])
            nxtV = nxtVec[idx]  # posV: 1*30   nxtV:1*30
            negV = negVec[idx]  # negV: 5(neg_num)*30
            Lcap_nxt = torch.sigmoid(nxtV.mm(posV.transpose(0, 1)))
            Lcap_neg = torch.sigmoid(negV.mm(posV.transpose(0, 1)))
            for i, lc in enumerate(Lcap_nxt):
                crsetploss_idx = -lc * torch.log(q_value + 1e-13) - (1-lc) * torch.log(1 - q_value + 1e-13)
            for i, ln in enumerate(Lcap_neg):
                crsetploss_idx += -ln * torch.log(q_value_neg[i] + 1e-13) - (1-ln) * torch.log(1 - q_value_neg[i] + 1e-13)
            crsetploss[idx].append(crsetploss_idx)
        return crsetploss   #batchsize*1

    def newCrsEtpLossFun(t_pos, t_nxt, t_neg, topic_dis_pos, topic_dis_nxt):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = topic_represention_V(t_nxt)
        crsetploss = [[] for _ in xrange(batch_size)]
        for idx, posV in enumerate(posVec):
            q_value = torch.index_select(topic_dis_pos[idx],0,t_pos[idx])  #topic_distribution[idx]: 1*25(topic_num), t_pos[idx]: 1
            q4nxtvalue = torch.index_select(topic_dis_pos[idx],0,t_nxt[idx])
            qnxtvalue = torch.index_select(topic_dis_nxt[idx],0,t_nxt[idx])
            q_value_neg = torch.index_select(topic_dis_pos[idx],0,t_neg[idx])
            nxtV = nxtVec[idx]  # posV: 1*30   nxtV:1*30
            negV = negVec[idx]  # negV: 5(neg_num)*30
            Lcap_nxt = torch.sigmoid(nxtV.mm(posV.transpose(0, 1)))
            Lcap_neg = torch.sigmoid(negV.mm(posV.transpose(0, 1)))
            crsetploss_idx = -q4nxtvalue * torch.log(qnxtvalue + 1e-13) - (1 - q4nxtvalue) * torch.log(1 - qnxtvalue + 1e-13)
            for i, lc in enumerate(Lcap_nxt):
                if t_pos[idx]!=t_nxt[idx]:
                    crsetploss_idx += -lc * torch.log(q_value + 1e-13) - (1-lc) * torch.log(1 - q_value + 1e-13)
            for i, ln in enumerate(Lcap_neg):
                crsetploss_idx += -ln * torch.log(q_value_neg[i] + 1e-13) - (1-ln) * torch.log(1 - q_value_neg[i] + 1e-13)
            crsetploss[idx].append(crsetploss_idx)
        return crsetploss   #batchsize*1

    def newnewCrsEtpLossFun(t_pos, t_nxt, t_neg, topic_dis_pos, topic_dis_nxt):
        posVec = topic_represention_U(t_pos)
        negVec = topic_represention_V(t_neg)
        nxtVec = topic_represention_V(t_nxt)
        crsetploss = [[] for _ in xrange(batch_size)]
        for idx, posV in enumerate(posVec):
            q_value = torch.index_select(topic_dis_pos[idx],0,t_nxt[idx])  #topic_distribution[idx]: 1*25(topic_num), t_pos[idx]: 1
            qself_value = torch.index_select(topic_dis_pos[idx], 0, t_pos[idx])
            q4nxtvalue = torch.index_select(topic_dis_pos[idx],0,t_nxt[idx])
            qnxtvalue = torch.index_select(topic_dis_nxt[idx],0,t_pos[idx])
            q_value_neg = torch.index_select(topic_dis_pos[idx],0,t_neg[idx])
            nxtV = nxtVec[idx]  # posV: 1*30   nxtV:1*30
            negV = negVec[idx]  # negV: 5(neg_num)*30
            Lcap_nxt = torch.sigmoid(nxtV.mm(posV.transpose(0, 1)))
            Lcap_neg = torch.sigmoid(negV.mm(posV.transpose(0, 1)))
            # crsetploss_idx = -torch.log(1-qself_value+1e-13)#-qnxtvalue * torch.log(q4nxtvalue + 1e-13) - (1 - qnxtvalue) * torch.log(1 - q4nxtvalue + 1e-13)
            crsetploss_idx = 0
            for i, lc in enumerate(Lcap_nxt):
                # if t_pos[idx]!=t_nxt[idx]:
                crsetploss_idx += -lc * torch.log(q_value + 1e-13) - (1-lc) * torch.log(1 - q_value + 1e-13)
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
            filelist = postopic.ListFile(corpus_path)
            filelist = postopic.BatchListFile(filelist, 64)
            loss1 = 0.
            loss2 = 0.
            for file in filelist:
                cur_win, nxt_win = postopic.BatchshortWholeWindowPick(file, winsiz, stride=1)
                batch_size = len(cur_win)
                # winsiz_list = [winsiz for _ in xrange(batch_size)]
                # cur_win, nxt_win, y_topic = postopic.BatchRMPick(batch_size, corpus_path, winsiz_list)
                pos_context_emb = pospart.load_wordembed(cur_win)
                nxt_context_emb = pospart.load_wordembed(nxt_win)
                # pos_context_emb = w2v_model.normalize(pos_context_emb)
                # nxt_context_emb = w2v_model.normalize(nxt_context_emb)
                pos_topic_dis, pos_sf_dis = move_forward(pospart, pos_context_emb)
                nxt_topic_dis, nxt_sf_dis = move_forward(negpart, nxt_context_emb)
                pos_temp_list = []
                pred_topic = torch.max(pos_topic_dis,1)[1]
                t_pos = sample_topic(pos_sf_dis)
                # t_pos = Variable(torch.LongTensor(pos_temp_list)).cuda(using_gpu) if cuda_valid else Variable(
                #     torch.LongTensor(pos_temp_list))
                nxt_temp_list = []
                t_nxt = sample_topic(nxt_sf_dis)
                # t_nxt = Variable(torch.LongTensor(nxt_temp_list)).cuda(using_gpu) if cuda_valid else Variable(
                #     torch.LongTensor(nxt_temp_list))
                t_neg = Variable(torch.LongTensor(sample_neg(t_pos, t_nxt))).cuda(using_gpu) if cuda_valid else Variable(torch.LongTensor(sample_neg(t_pos, t_nxt)))
                logLoss = LogLossFun(t_pos, t_nxt, t_neg)
                optimizer_U.zero_grad()
                optimizer_V.zero_grad()
                for item in logLoss:
                    item.backward(retain_variables=True)
                    loss1 += item
                optimizer_U.step()
                optimizer_V.step()
                crsetcrstepVal = CrsEtpLossFun(t_pos, t_nxt, t_neg, pos_topic_dis)
                optimizer_pos.zero_grad()
                for crete in crsetcrstepVal:
                    for item in crete:
                        for i in item:
                            loss2 += i
                            i.backward(retain_variables=True)
            loss1 = loss1/len(filelist)
            loss2 = loss2/len(filelist)
            print loss2
            info = {
                'loss1': loss1.data[0],
                'loss2': loss2.data[0]
            }
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
        torch.save(pospart.state_dict(), 'model/20newsmuse818/pospartmuse' + str(iter))
        torch.save(topic_represention_U, 'model/20newsmuse818/rU'+str(iter))
        torch.save(topic_represention_V, 'model/20newsmuse818/rV'+str(iter))
        exit()
    torch.save(pospart.state_dict(), 'model/20newsmuse818/pospartmuse' + str(iter))
    torch.save(topic_represention_U, 'model/20newsmuse818/rU'+str(iter))
    torch.save(topic_represention_V, 'model/20newsmuse818/rV'+str(iter))

if __name__ == '__main__':
    main()
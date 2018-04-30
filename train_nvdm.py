import torch
import torch.nn as nn
from torch.autograd import Variable
import model.w2vloader as w2vld
import model.utils as postopic
import random
import time
import numpy as np
from torch import optim
import model.hdrl4tm as hdrl4tm
from model.logger import Logger
import readdata

def main():
    iter = 0
    MaxIteration = 1000
    topic_num = 50
    word_num = 3089
    emb_dim = 50
    epsilon = 0.8
    topic_dim = 300
    using_gpu = 0
    batch_size = 64
    learning_rate = 5e-4
    hidden_size = 500
    meta_c = hdrl4tm.nvdm_encoder(word_num, hidden_size, topic_num).cuda(using_gpu)
    ctr = hdrl4tm.nvdm_ctr(word_num, emb_dim, topic_num, using_gpu).cuda(using_gpu)
    # meta_c.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newnvdm817/test/meta999'))
    # ctr.load_state_dict(torch.load('/home/shiyi/rlfortopicmodel/model/20newnvdm817/test/ctr999'))
    logger = Logger('./logger/reuterssnvdm101')
    opt_meta = optim.Adam(meta_c.parameters(), learning_rate)
    opt_ctr = optim.Adam(ctr.parameters(), learning_rate)
    torch.manual_seed(0)
    def to_np(x):
        return x.data.cpu().numpy()

    try:
        for iter in range(MaxIteration):
            data_url = "reutersbag.txt"
            batches = readdata.CreateBatches(data_url, batch_size, shuffle=True)
            # batches = batches[-2:]
            total_kld = 0.
            sfm = torch.nn.LogSoftmax()
            total_e_dis = 0.
            for b_idx, batch in enumerate(batches):
                for ende in range(2):
                    batchwordlist, batchwordoccur = readdata.PickInBatch(data_url, batch, batch_size, word_num, mode=1) #64*10000/2000 64*words
                    goal_mean, goal_logsig = meta_c(Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu))
                    test = goal_mean* goal_mean
                    kld = torch.sum(-0.5*(1 - goal_mean*goal_mean + 2 * goal_logsig - torch.exp(2 * goal_logsig)), 1)
                    EPS = Variable(torch.normal(torch.zeros(len(batchwordlist),topic_num), torch.ones(len(batchwordlist),topic_num))).cuda(using_gpu)
                    goal = goal_mean + torch.exp(goal_logsig) * EPS
                    topic_dis = ctr(goal)
                    batchwordlist = Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu)
                    logits = sfm(topic_dis)
                    test = torch.sum(logits, 1)
                    recon_loss = -torch.sum(batchwordlist * logits ,1)
                    total_e_dis += torch.sum(recon_loss)
                    opt_meta.zero_grad()
                    for kld_val in kld:
                        kld_val.backward(retain_variables=True)
                    total_kld += torch.sum(kld)
                    opt_ctr.zero_grad()
                    for loss_val in recon_loss:
                        loss_val.backward(retain_variables=True)
                    if ende == 0:
                        opt_meta.step()
                    else:
                        opt_ctr.step()
            info = {
                # 'accuracy': avg_accuracy
                'kld': to_np(total_kld/(2*len(batches))),
                'total_loss' : to_np(total_e_dis/(2*len(batches)))
            }
            print("epoch: %d || kld: %f || e-dis: %f" % (iter, to_np(total_kld/(2*len(batches))), to_np(total_e_dis/(2*len(batches)))))
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter + 1)

            for tag, value in meta_c.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), iter + 1)
                logger.histo_summary(tag + '/grad', to_np(value.grad), iter + 1)

            for tag, value in ctr.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), iter + 1)
                logger.histo_summary(tag + '/grad', to_np(value.grad), iter + 1)

    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'
        torch.save(meta_c.state_dict(), 'model/reuterssnvdm103/meta' + str(iter))
        torch.save(ctr.state_dict(), 'model/reuterssnvdm103/ctr' + str(iter))
        exit()
    torch.save(meta_c.state_dict(), 'model/reuterssnvdm103/meta' + str(iter))
    torch.save(ctr.state_dict(), 'model/reuterssnvdm103/ctr' + str(iter))

if __name__ == '__main__':
    main()
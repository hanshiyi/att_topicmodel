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

def main():
    iter = 0
    MaxIteration = 1000
    topic_num = 50
    word_num = 2000
    emb_dim = 500
    using_gpu = 1
    batch_size = 8
    learning_rate = 0.001
    vocab_file = 'data/vocab.new'
    ctr = hdrl4tm.transitionmodel(word_num, emb_dim, topic_num, vocab_file, using_gpu).cuda(using_gpu)
    w2idx = dict()
    w2idx[''] = 0
    with open(vocab_file) as file:
        idxx = 1
        for line in file.readlines():
            w2idx[line.strip().split()[0]] = idxx
            idxx += 1
    logger = Logger('./logger/20newsgroup0429')
    opt_ctr = optim.Adam(ctr.parameters(), learning_rate)
    def to_np(x):
        return x.data.cpu().numpy()

    try:
        for iter in range(MaxIteration):
            destr = "data/trainlist.txt"
            bag_url = "data/trainbag2000.txt"
            filelist, randomfilelist = postopic.FileListfromFile(destr)
            batchfilelist = postopic.BatchListFile(randomfilelist, batch_size)
            baglist = postopic.ListBag(bag_url)
            # batches = readdata.CreateBatches(data_url, batch_size, shuffle=True)
            # batches = batches[-2:]
            total_kld = 0.
            total_e_dis = 0.
            sig = torch.nn.Sigmoid()
            sfm = torch.nn.LogSoftmax()
            for b_idx, batch in enumerate(batchfilelist):
                winlst, batchwordlist = postopic.BatchPickWindowFromFile(batch, baglist, filelist, word_num, w2idx)
                topic_dis, kld = ctr(winlst, Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu))
                batchwordlist = Variable(torch.FloatTensor(batchwordlist)).cuda(using_gpu)
                logits = sfm(topic_dis)
                recon_loss = -torch.sum(batchwordlist * logits, 1)
                total_e_dis += torch.sum(recon_loss)
                opt_ctr.zero_grad()
                for kld_val in kld:
                    kld_val.backward(retain_variables=True)
                total_kld += torch.sum(kld)
                for loss_val in recon_loss:
                    loss_val.backward(retain_variables=True)
            info = {
                # 'accuracy': avg_accuracy
                'kld': to_np(total_kld),
                'e-distance' : to_np(total_e_dis)
            }
            print("epoch: %d || kld: %f || e-dis: %f" % (iter, to_np(total_kld), to_np(total_e_dis)))
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iter + 1)

            for tag, value in ctr.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), iter + 1)
                logger.histo_summary(tag + '/grad', to_np(value.grad), iter + 1)

    except KeyboardInterrupt:
        print 'termination by ' + str(iter) + 'iterations'
        torch.save(ctr.state_dict(), 'model/20news0429/ctr' + str(iter))
        exit()
    torch.save(ctr.state_dict(), 'model/20news0429/ctr' + str(iter))

if __name__ == '__main__':
    main()

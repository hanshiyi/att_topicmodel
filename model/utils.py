import torch
import torch.nn as nn
import random
import os
from os.path import join
import time
import numpy as np
from readdata import CreateBatches as cb
from readdata import PickInBatch as pb


def pick(rootdest, neg_num):
    topicList = []
    posData = []
    negDataS = [[] for _ in xrange(neg_num)]
    oldIdx = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)

    random.seed(time.time())
    randIdx = random.randint(0, len(topicList) - 1)
    tempTopic = topicList[randIdx]
    posDataDir = join(rootdest, tempTopic)

    for root, dirs, files in os.walk(posDataDir):
        file_names = files
    x = random.randint(0, len(file_names) - 1)

    for line in open(join(posDataDir, file_names[x])):
        for st in line.split():
            posData.append(st)

    for i in range(neg_num):
        randIdx = random.randint(0, len(topicList) - 1)
        while randIdx in oldIdx:
            randIdx = random.randint(0, len(topicList) - 1)
        oldIdx.append(randIdx)

        tempTopic = topicList[randIdx]
        negDataDir = join(rootdest, tempTopic)
        for root, dirs, files in os.walk(negDataDir):
            file_names = files
        x = random.randint(0, len(file_names) - 1)

        for line in open(join(negDataDir, file_names[x])):
            for st in line.split():
                negDataS[i].append(st)

    negLen = []
    for negData in negDataS:
        negLen.append(negData.__len__())

    negValidLen = min(negLen)

    posValid = posData
    negValidS = [[] for _ in xrange(neg_num)]
    for i in range(neg_num):
        negValidS[i] = negDataS[i][0:negValidLen-1]

    return posValid, negValidS


def WindowPick(rootdest, window_size):
    topicList = []
    data = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    # print topicList

    random.seed(time.time())
    randIdx = []
    randIdx.append(random.randint(0, len(topicList) - 1))
    posDataDir = join(rootdest, topicList[randIdx[0]])

    for root, dirs, files in os.walk(posDataDir):
        file_names = files
    x = random.randint(0, len(file_names) - 1)

    for line in open(join(posDataDir, file_names[x])):
        for st in line.split():
            data.append(st)

    first_idx = random.randint(0, len(data) - 2*window_size - 2)
    surr_word = []
    n_surr_word = []
    displace = random.randint(1, window_size)
    # displace = max(displace, -first_idx)
    displace = min(displace, len(data)-1-first_idx- 2*window_size-1)
    for i in range(window_size * 2 + 1):
        surr_word.append(data[first_idx + i])
        n_surr_word.append(data[first_idx + i + displace])
    return surr_word, n_surr_word, randIdx


def ListFile(rootdest, shuffle=True):
    filelist = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            destr = join(rt, OneDirName)
            for root, dirs, files in os.walk(destr):
                for OneFileName in files:
                    filer = join(destr, OneFileName)
                    filelist.append(filer)
    if shuffle:
        random.shuffle(filelist)
    return filelist


def BatchListFile(filelist, batch_size):
    batches = [[] for _ in xrange(len(filelist)/batch_size+1)]
    for batch_idx in range(len(filelist)/batch_size):
        batches[batch_idx] = filelist[batch_idx*batch_size:(batch_idx+1)*batch_size]
    batches[len(filelist)/batch_size] = filelist[len(filelist)/batch_size*batch_size:]
    return batches


def WholeWindowPick(file, window_size, stride=1):
    cur_batchwordlist = []
    nxt_batchwordlist = []
    data = []
    for line in open(file):
        for st in line.split():
            data.append(st)
    window = [[] for _ in xrange(len(data) - window_size +1)]
    for first_idx in range(len(data) - window_size + 1):
        for j in range(window_size):
            window[first_idx].append(data[first_idx+j])
    for j in range((len(data) - window_size + 1)/stride):
        first_idx = j*stride
        cur_batchwordlist.append(window[first_idx])
        displace = 0
        while displace == 0:
            displace = random.randint(-window_size + 1, window_size - 1)
            displace = max(displace, -first_idx)
            displace = min(displace, len(data) - 1 - first_idx - window_size)
        nxt_batchwordlist.append(window[first_idx + displace])
    return cur_batchwordlist, nxt_batchwordlist

def shortWholeWindowPick(file, window_size, stride=1):
    cur_batchwordlist = []
    nxt_batchwordlist = []
    data = []
    for line in open(file):
        for st in line.split():
            data.append(st)
    first_idx = random.randint(0, len(data) - window_size-1)
    surr_word = []
    n_surr_word = []
    displace = random.randint(1, window_size/2)
    # displace = max(displace, -first_idx)
    displace = min(displace, len(data) - 1 - first_idx - window_size)
    for i in range(window_size):
        surr_word.append(data[first_idx + i])
        n_surr_word.append(data[first_idx + i + displace])
    cur_batchwordlist.append(surr_word)
    nxt_batchwordlist.append(n_surr_word)
    return cur_batchwordlist, nxt_batchwordlist


def BatchshortWholeWindowPick(batchfile, window_size, stride=1):
    cur_batchwordlist = [[] for _ in xrange (len(batchfile))]
    nxt_batchwordlist = [[] for _ in xrange (len(batchfile))]
    for idx, file in enumerate(batchfile):
        data = []
        for line in open(file):
            for st in line.split():
                data.append(st)
        first_idx = random.randint(0, len(data) - window_size - 1)
        surr_word = []
        n_surr_word = []
        displace = random.randint(1, window_size -1)
        displace = (-1)**random.randint(0,1) * displace
        displace = max(displace, -first_idx)
        displace = min(displace, len(data) - 1 - first_idx - window_size)
        for i in range(window_size):
            surr_word.append(data[first_idx + i])
            n_surr_word.append(data[first_idx + i + displace])
        cur_batchwordlist[idx] = surr_word
        nxt_batchwordlist[idx] = n_surr_word
    return cur_batchwordlist, nxt_batchwordlist


def FileListfromFile(file, shuffle=True):
    flist = open(file)
    flines = flist.readlines()
    flist.close()
    filelist = []
    randomfilelist = []
    for fline in flines:
        filelist.append(fline.split()[0])
        randomfilelist.append(fline.split()[0])
    if shuffle:
        random.shuffle(randomfilelist)
    return filelist, randomfilelist


def ListBag(bag_url):
    blist = open(bag_url)
    blines = blist.readlines()
    blist.close()
    baglist = []
    for bline in blines:
        baglist.append(bline)
    return baglist


def BatchPickWindowFromFile(batchfilelist, baglist, filelist, word_num, w2idx):
    winlist = [[] for _ in xrange(len(batchfilelist))]
    batchwordlist = [[0 for _ in xrange(word_num)] for _ in xrange(len(batchfilelist))]

    for idx, file in enumerate(batchfilelist):
        data = []
        # for line in open(join('..',file)):
        for line in open(file):
            for st in line.split():
                data.append(st)
        # for i in range(len(data)/window_size):
        winlist[idx] = data
        list_num = filelist.index(file)
        wfs = baglist[list_num].split()
        for i in range(1, len(wfs)):
            w = int(wfs[i].split(":")[0])
            f = int(wfs[i].split(":")[1])
            batchwordlist[idx][w] = f
    dwinlist = sorted(winlist, key=lambda lst:len(lst), reverse=True)
    clipedtext = []
    cliped_len = []
    for do in dwinlist:
        if len(do) > 100:
            J_step = len(do) / 100
            clipedtext.append(do[::J_step][:100])
        else:
            clipedtext.append(do)
        cliped_len.append(len(clipedtext[-1]))
        if cliped_len[-1] < cliped_len[0]:
            for _ in range(cliped_len[0] - cliped_len[-1]):
                clipedtext[-1].append('')
        clipedtext[-1] = [w2idx[i] for i in clipedtext[-1]]
    decbatchlist = sorted(batchwordlist, key=lambda batch:np.sum(batch), reverse=True)
    return [clipedtext, cliped_len], decbatchlist

def newBatchPickWindowFromFile(batchfilelist, baglist, filelist, word_num, wordfile):
    winlist = [[] for _ in xrange(len(batchfilelist))]
    batchwordlist = [[0 for _ in xrange(word_num)] for _ in xrange(len(batchfilelist))]
    wordlist=[]
    for line in open(wordfile,'r').readlines():
        wordlist.append(line.strip())
    for idx, file in enumerate(batchfilelist):
        data = []
        # for line in open(join('..',file)):
        for line in open(file):
            for st in line.split():
                if st in wordlist:
                    data.append(st)
        # for i in range(len(data)/window_size):
        winlist[idx] = data
        list_num = filelist.index(file)
        wfs = baglist[list_num].split()
        for i in range(1, len(wfs)):
            w = int(wfs[i].split(":")[0])
            f = int(wfs[i].split(":")[1])
            batchwordlist[idx][w] = f
    dwinlist = sorted(winlist, key=lambda lst:len(lst), reverse=True)
    clipedtext = []
    for do in dwinlist:
        if len(do) > 50:
            J_step = len(do) / 50
            clipedtext.append(do[::J_step][:50])
        else:
            clipedtext.append(do)
    decbatchlist = sorted(batchwordlist, key=lambda batch:np.sum(batch), reverse=True)
    return clipedtext, decbatchlist


def randomWindowPick(rootdest, window_size):
    topicList = []
    data = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    # print topicList

    random.seed(time.time())
    randIdx = []
    randIdx.append(random.randint(0, len(topicList) - 1))
    posDataDir = join(rootdest, topicList[randIdx[0]])

    for root, dirs, files in os.walk(posDataDir):
        file_names = files
    x = random.randint(0, len(file_names) - 1)

    for line in open(join(posDataDir, file_names[x])):
        for st in line.split():
            data.append(st)

    first_idx = random.randint(0, len(data) - 2 * window_size - 2)
    surr_word = []
    for i in range(window_size * 2 + 1):
        surr_word.append(data[first_idx + i])

    first_idx = random.randint(0, len(data) - 2 * window_size - 2)
    temp_word = []
    for i in range(window_size * 2 + 1):
        temp_word.append(data[first_idx + i])
    return surr_word, temp_word, randIdx

def MiniBagPick(rootdest, window_size, bag_num):
    topicList = []
    data = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    # print topicList

    random.seed(time.time())
    randIdx = []
    randIdx.append(random.randint(0, len(topicList) - 1))
    posDataDir = join(rootdest, topicList[randIdx[0]])

    for root, dirs, files in os.walk(posDataDir):
        file_names = files
    x = random.randint(0, len(file_names) - 1)

    for line in open(join(posDataDir, file_names[x])):
        for st in line.split():
            data.append(st)

    first_idx = random.randint(0, len(data) - 2*window_size - 2)
    surr_word = []
    for i in range(window_size * 2 + 1):
        surr_word.append(data[first_idx + i])

    mini_bag_word = []
    for times in xrange(bag_num):
        first_idx = random.randint(0, len(data) - 2 * window_size - 2)
        temp_word = []
        for i in range(window_size * 2 + 1):
            temp_word.append(data[first_idx + i])
        mini_bag_word.append(temp_word)
    return surr_word, mini_bag_word, randIdx

def BatchPick(batch_size, destr, wndsiz_list):
    cur_wrdlstBatch = []
    nxt_wrdlstBatch = []
    topicidxBatch = []
    for i in range(batch_size):
        cur_wrdlst, nxt_wrdlst, topicidx = WindowPick(destr, wndsiz_list[i])
        cur_wrdlstBatch.append(cur_wrdlst)
        nxt_wrdlstBatch.append(nxt_wrdlst)
        topicidxBatch.append(topicidx)
    return cur_wrdlstBatch, nxt_wrdlstBatch, topicidxBatch

def BatchRMPick(batch_size, destr, wndsiz_list):
    cur_wrdlstBatch = []
    nxt_wrdlstBatch = []
    topicidxBatch = []
    for i in range(batch_size):
        cur_wrdlst, nxt_wrdlst, topicidx = randomWindowPick(destr, wndsiz_list[i])
        cur_wrdlstBatch.append(cur_wrdlst)
        nxt_wrdlstBatch.append(nxt_wrdlst)
        topicidxBatch.append(topicidx)
    return cur_wrdlstBatch, nxt_wrdlstBatch, topicidxBatch

def BatchMBPick(batch_size, destr, wndsiz_list, bag_num):
    cur_wrdlstBatch = []
    nxt_wrdlstBatch = []
    topicidxBatch = []
    for i in range(batch_size):
        cur_wrdlst, nxt_wrdlst, topicidx = MiniBagPick(destr, wndsiz_list[i], bag_num)
        cur_wrdlstBatch.append(cur_wrdlst)
        nxt_wrdlstBatch.append(nxt_wrdlst)
        topicidxBatch.append(topicidx)
    return cur_wrdlstBatch, nxt_wrdlstBatch, topicidxBatch

def TestPick(rootdest, window_size):
    wordList = []
    topicList = []
    topicIdx = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    # print topicList
    for idx, topic in enumerate(topicList):
        txtDir = join(rootdest, topic)
        for root, dirs, files in os.walk(txtDir):
            file_names = files
            for file in file_names:
                data = []
                for line in open(join(txtDir, file)):
                    for st in line.split():
                        data.append(st)
                first_idx = random.randint(0, len(data) - 2 * window_size - 2)
                surr_word = []
                for i in range(window_size * 2 + 1):
                    surr_word.append(data[first_idx + i])
                wordList.append(surr_word)
                topicIdx.append(idx)
    return wordList, topicIdx

def TestPickAll(rootdest):
    wordList = []
    topicList = []
    topicIdx = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    for idx, topic in enumerate(topicList):
        txtDir = join(rootdest, topic)
        for root, dirs, files in os.walk(txtDir):
            file_names = files
            for file in file_names:
                data = []
                for line in open(join(txtDir, file)):
                    for st in line.split():
                        data.append(st)
                wordList.append(data)
                topicIdx.append(idx)
    return wordList, topicIdx

def ListTopic(rootdest, destw):
    topicList = []
    for rt, drs, fls in os.walk(rootdest):
        for OneDirName in drs:
            topicList.append(OneDirName)
    if os.path.exists(destw) == False:
        os.makedirs(destw)
    f1 = open(join(destw, "metadata.tsv"), 'w')
    f1.write("ID\tTopic\n")
    for i in range(len(topicList)):
        f1.write("%d" %i)
        f1.write("\t")
        f1.write(topicList[i])
        f1.write("\n")




if __name__ == '__main__':
    neg_num = 4
    destr="../data/nipstrainlist.txt"
    bag_url = "../data/nipstrainbag2000.txt"
    window_size = 3
    batch_size = 128
    wordnum = 2000

    filelist, randomfilelist = FileListfromFile(destr)
    batchfilelist = BatchListFile(randomfilelist, batch_size)
    baglist = ListBag(bag_url)
    for batch in batchfilelist:
        winlst, batchwordlst = BatchPickWindowFromFile(batch, baglist, filelist, wordnum)
        pass

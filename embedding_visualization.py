import torch
import tensorflow as tf
import sys
import model.hdrl4tm as hdrl4tm
import torch.nn as nn
from tensorflow.contrib.tensorboard.plugins import projector
import os

def main():
    topic_num = 50
    word_num = 2000
    emb_dim = 50
    neg_num = 5
    pos_num = 0
    epsilon = 0.8
    topic_dim = 50
    using_gpu = 3
    cuda_valid = False
    file_dir = "/home/davidwang/rlfortopicmodel/"
    LOG_DIR = './logger/20news_embedding'
    # cuda_valid = False
    # topic_represention_U = nn.Embedding(topic_num, topic_dim).cuda(using_gpu) if cuda_valid else nn.Embedding(topic_num,                                                                                topic_dim)
    session = tf.Session()
    ctr = hdrl4tm.controler2(word_num, emb_dim, topic_num, 'vocab.new', using_gpu)
    ctr.load_state_dict(torch.load('/home/davidwang/rlfortopicmodel/model/20newshrl94/test/ctr999'))
    a = ctr.emb
    # topic_represention_U.load_state_dict(a)
    embedding_matrix = tf.constant_initializer(a.weight.data.cpu().numpy())


    # Create randomly initialized embedding weights which will be trained.
    embedding_var = tf.get_variable(name='topic_embedding', initializer=embedding_matrix, shape=[word_num, 50],
                                    dtype=tf.float32)
    session.run(embedding_var.initializer)
    saver = tf.train.Saver()
    saver.save(session, os.path.join(LOG_DIR, "embedding_20newshrl.ckpt"), 0)

    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(file_dir, 'meta.tsv')

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
    



if __name__ == '__main__':
    main()
import numpy as np
import tensorflow as tf

class CharCNN(object):
    def __init__(self,sequence_length,num_class,embedding_size,filter_size,vocab_size):
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None,num_class],name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')


        with tf.device('/gpu:0'),tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),trainable=False,name='W')
            self.embed_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            self.embed_chars_expanded = tf.expand_dims(self.embed_chars,-1)

        pooled_outputs = []
        for i,filter_sized in enumerate(filter_size):
            with tf.name_scope('conv-maxpool-%s'% filter_sized):
                filter_shape = [filter_sized,embedding_size,1,32]
                W = tf.Variable(tf.random_uniform(filter_shape,minval=-0.25,maxval=0.25),name='W')
                b = tf.Variable(tf.constant(0.1,shape=[32]),name='b')
                conv = tf.nn.conv2d(self.embed_chars_expanded,W,strides=[1,1,1,1],padding='VALID',name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
                pooled = tf.nn.max_pool(h,ksize=[1,4,1,1],strides=[1,1,1,1],padding='VALID',name='max-pool')
                pooled_outputs.append(pooled)

        with tf.name_scope('h_pool'):
            #total_num_filters = 32*len(filter_size)
            self.h_pool = tf.concat(pooled_outputs,3,name='h_pool_not_flat')
            self.h_pool_flat = tf.contrib.layers.flatten(self.h_pool)


        """Fully-Connected layer 1"""
        with tf.name_scope('fc1'):
            W1 = tf.Variable(tf.random_uniform([995*1*32,50],minval=-0.25,maxval=0.25),name='W1')
            b = tf.Variable(tf.constant(0.1,shape=[50]),name='b')
            self.fc1 = tf.nn.xw_plus_b(self.h_pool_flat,W1,b,name='fc1_output')

        with tf.name_scope('dropout_fc1'):
            self.fc1_drop = tf.nn.dropout(self.fc1,keep_prob=self.dropout_keep_prob)


        """Fully-Connected layer 2"""
        with tf.name_scope('fc2'):
            filter_shape = [50,50]
            W2 = tf.Variable(tf.random_uniform(filter_shape,minval=-0.25,maxval=0.25),name='W2')
            b = tf.Variable(tf.constant(0.1,shape=[50]),name='b')
            self.fc2 = tf.nn.xw_plus_b(self.fc1_drop,W2,b,name='fc2_output')

        with tf.name_scope('dropout_fc2'):
            self.fc2_drop = tf.nn.dropout(self.fc2,keep_prob=self.dropout_keep_prob)

        """Final output layer"""
        with tf.name_scope('output'):
            filter_shape = [50,1]
            W3 = tf.Variable(tf.random_uniform(filter_shape,minval=-0.25,maxval=0.25),name='W3')
            b = tf.Variable(tf.constant(0.1,shape=[num_class]),name='b')
            self.scores = tf.nn.xw_plus_b(self.fc2_drop,W3,b,name='scores')
            self.sigmoid = tf.sigmoid(self.scores,name='sigmoid')

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.sigmoid,self.input_y,name='squared_diff'))
            tf.summary.scalar('loss_',self.loss)

        self.merge = tf.summary.merge_all()

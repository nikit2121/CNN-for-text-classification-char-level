from __future__ import division
import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
import logging,json
import tensorflow as tf
from charCNN_toxic import CharCNN
import time,os,pickle
from tensorflow.python import debug as tf_debug
logging.getLogger().setLevel(logging.INFO)

train_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/train/train.csv')
maxlen = 1000
embed_size = 50

def create_vocab():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) +
                list(string.punctuation) + ['\n'])
    vocab_size = len(alphabet)
    check = set(alphabet)
    letter_to_id = {}
    id_to_letter = {}
    for index,letter in enumerate(alphabet):
        letter_to_id[letter] = index
        id_to_letter[index] = letter
    return letter_to_id,id_to_letter,vocab_size,check



def word_to_letters(s):
    s = s.lower()
    s = "".join([i for i in s if i in string.printable])#another try to remove non-ascii characters since the one below didnt work
    s = re.sub(' ','',s)
    s = re.sub(r'[^\x00-\x7f]+',r'',s)#remove-ascii codes
    s = [letter for letter in s]

    return s

def encode_data(x_raw,letter_to_id,maxlen):
    sentence_array = np.zeros((len(x_raw),maxlen),np.int)
    for i,sent in enumerate(x_raw):
        count = 0
        for j,value in enumerate(sent):
            if count>=maxlen:
                pass
            else:
                sentence_array[i,j] = letter_to_id[value]
                count+=1
    return sentence_array

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """Iterate the data batch by batch"""
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size)+1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]



letter_to_id,id_to_letter,vocab_size,check = create_vocab()
x_raw = train_data['comment_text'].apply(lambda x: word_to_letters(x)).tolist()
x = encode_data(x_raw,letter_to_id,maxlen)
y = np.array(train_data['toxic'].tolist())
y = y.transpose()
y = np.reshape(y,(len(y),1))

shuffle_indices = np.random.permutation(np.arange(len(y)))
x = x[shuffle_indices]
y = y[shuffle_indices]

#x_,x_test,y_,y_test = train_test_split(x,y,test_size=0.01,random_state=45)
x_train,x_dev,y_train,y_dev = train_test_split(x,y,test_size=0.01,random_state=45)

logging.info('x_train: {}, x_dev: {}' .format(len(x_train),len(x_dev)))
logging.info('y_train: {}, y_dev: {}' .format(len(y_train),len(y_dev)))

parameter = json.loads(open('/home/nikit/Desktop/Kaggle/toxic_comments_character_level/parameters_toxic.json').read())
graph = tf.Graph()
with graph.as_default():
    session_Conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    sess = tf.Session(config=session_Conf)
    with sess.as_default():
        charCNN = CharCNN(sequence_length=x_train.shape[1],
                          num_class=1,
                          embedding_size=parameter['embedding_dim'],
                          filter_size=list(map(int, parameter['filter_sizes'])),
                          vocab_size=vocab_size)
                          #num_filters=parameter['num_filters'])
        global_step = tf.Variable(0,name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-5)
        grads_and_vars = optimizer.compute_gradients(charCNN.loss)
        train_op = optimizer.apply_gradients(grads_and_vars,global_step=global_step)
        timestamp = str(int(time.time()))


        main_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_"+timestamp))
        checkpoint_dir = os.path.abspath(os.path.join(main_dir, "checkpoints"))
        checkpoint_prefix = os.path.abspath(os.path.join(checkpoint_dir, "model"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())
        vocab_file = open(os.path.join(main_dir, "vocab.pickle"), 'wb')
        pickle.dump(letter_to_id, vocab_file)
        vocab_file.close()

        sess.run(tf.global_variables_initializer())

        train_batches = batch_iter(list(zip(x_train, y_train)), parameter['batch_size'], parameter['num_epochs'])
        best_loss, best_at_step,average_dev_loss = 1000,0,0
        file_writer = tf.summary.FileWriter('/home/nikit/Desktop/Kaggle/toxic_comments_character_level/', sess.graph)
        def train_step(x_batch, y_batch):
            feed_dict = {charCNN.input_x: x_batch, charCNN.input_y: y_batch, charCNN.dropout_keep_prob: parameter['embedding_dim']}
            _, step, loss,summary = sess.run([train_op, global_step, charCNN.loss,charCNN.merge], feed_dict=feed_dict)
            file_writer.add_summary(summary,step)
            return step, loss

        def dev_step(x_batch, y_batch):
            feed_dict = {charCNN.input_x: x_batch, charCNN.input_y: y_batch, charCNN.dropout_keep_prob: 1}
            _, step, loss = sess.run([train_op, global_step, charCNN.loss], feed_dict=feed_dict)
            return loss
        total_length = 0
        for train_batch in train_batches:
            x_train_batch, y_train_batch = zip(*train_batch)
            step_, train_loss = train_step(x_train_batch,y_train_batch)
            current_step = tf.train.global_step(sess,global_step)
            total_length+=len(x_train_batch)
            if int(total_length/parameter['batch_size'])==308:
                count, dev_total_loss = 1,0
                dev_batches = batch_iter(list(zip(x_dev, y_dev)), parameter['batch_size'], 1)
                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    dev_loss = dev_step(x_dev_batch,y_dev_batch)
                    dev_total_loss += dev_loss
                    count += 1
                average_dev_loss = dev_total_loss/count
                logging.critical("Average loss on dev set : {}".format(average_dev_loss))
                total_length = 0

                if average_dev_loss<=best_loss:
                    best_at_step = current_step
                    best_loss = average_dev_loss
                    path = saver.save(sess, checkpoint_prefix,global_step=current_step)
                    logging.critical('Saved model at {} at step {}'.format(path, best_at_step))
                    logging.critical('Best loss is {} at step {}'.format(best_loss, best_at_step))

        """
        test_batches = batch_iter(list(zip(x_test, y_test)), parameter['batch_size'], 1)
        count, total_loss = 1,0
        for test_batch in test_batches:
            x_test_batch, y_test_batch = zip(*test_batch)
            test_loss = dev_step(x_test_batch, y_test_batch)
            total_loss += test_loss
            count += 1
        average_test_loss = total_loss/count
        logging.critical("Average loss on test: {}".format(average_test_loss))"""
    sess.close()
uni, counts = np.unique(y_dev,return_counts=True)
print counts
print('hello')

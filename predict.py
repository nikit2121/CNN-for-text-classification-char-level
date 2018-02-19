import tensorflow as tf
import pandas as pd
import numpy as np
import string
import re,pickle,os,json
import logging
logging.getLogger().setLevel(logging.INFO)
test_data = pd.read_csv('/home/nikit/Desktop/Kaggle/toxic_comments/data/test/test.csv')
maxlen = 1000

def word_to_letters(s):
    s = s.lower()
    s = "".join([i for i in s if i in string.printable])#another try to remove non-ascii characters since the one below didnt work
    s = re.sub(' ','',s)
    s = re.sub('\t','',s)
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

def predict():
    parameter = json.loads(open('/home/nikit/Desktop/Kaggle/toxic_comments_character_level/parameters_toxic.json').read())
    model_dir = '/home/nikit/Desktop/Kaggle/toxic_comments_character_level/trained_model_1517622322/'
    if not model_dir.endswith('/'):
        model_dir+='/'
    file = open(os.path.join(model_dir,'vocab.pickle'),'rb')
    letter_to_id = pickle.load(file)

    x_raw = test_data['comment_text'].apply(lambda x: word_to_letters(x)).tolist()
    x_test = encode_data(x_raw,letter_to_id,maxlen)
    logging.info('Transformed test_data to their ids and length: {}'.format(len(x_test)))
    checkpoint_file = tf.train.latest_checkpoint(model_dir+'checkpoints')
    logging.critical('load trained model {}'.format(checkpoint_file))

    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        sess = tf.Session(config=session_config)

        with sess.as_default():
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess,checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            sigmoid = graph.get_operation_by_name("output/sigmoid").outputs[0]

            x_test_batches = batch_iter(list(x_test),parameter['batch_size'],1)
            all_scores = []
            for x_test_batch in x_test_batches:
                b_s = sess.run(sigmoid, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                for index,i in enumerate(b_s):
                    all_scores.append(i)
        sess.close()
    np.savetxt('/home/nikit/Desktop/Kaggle/toxic_comments/results/seventh/predictions6.csv',all_scores)
if __name__ == '__main__':
    predict()

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

root = "/Knowledge Discovery and Data Mining/project"
train_path = os.path.join(root, 'train/train')
train_labels = os.path.join(root, 'train_kaggle.csv')
test_path = os.path.join(root, 'test/test')


def data_generation(load):
    if load:
        with open("training_data.pkl", "rb") as pkl_f:
            train_dict = pickle.load(pkl_f)
        with open("testing_data.pkl", "rb") as pkl_f:
            test_dict = pickle.load(pkl_f)
        return train_dict, test_dict

    with open(train_labels, "rb") as label_f:
        lables = np.loadtxt(label_f, delimiter=",", skiprows=1)

    seq_max_len = 70
    dataset = {'train': [], 'seq_len': [], 'labels': []}

    for dirName, subdirList, fileList in os.walk(train_path):
        for fname in tqdm(fileList):
            sample_path = os.path.join(train_path, fname)
            id = np.squeeze(np.where(lables[:, 0] == int(fname.split('.')[0])))
            dataset['labels'].append(lables[id][1])
            x = np.load(sample_path)
            x = x[0:min(seq_max_len, x.shape[0])]
            if x.shape[0] > seq_max_len:
                x = x[:seq_max_len, :]
            x = np.nan_to_num(x)
            dataset['seq_len'].append(len(x))
            x = np.pad(x, ((0, seq_max_len - len(x)), (0, 0)), 'constant', constant_values=0)
            dataset['train'].append(x)

    dataset['train'] = np.array(dataset['train'])
    dataset['labels'] = np.expand_dims(np.array(dataset['labels']), axis=1)
    dataset['seq_len'] = np.expand_dims(np.array(dataset['seq_len']), axis=1)
    print(dataset['train'].shape, dataset['labels'].shape, dataset['seq_len'].shape)

    # ./lstm_xgboost/training_data.pkl
    with open("training_data.pkl", "wb") as pkl_f:
        pickle.dump(dataset, pkl_f, protocol=4)

    dataset = {'test': [], 'seq_len': [], 'id': []}
    for dirName, subdirList, fileList in os.walk(test_path):
        for fname in tqdm(fileList):
            sample_path = os.path.join(test_path, fname)
            dataset['id'].append(int(fname.split('.')[0]))
            x = np.load(sample_path)
            if x.shape[0] > seq_max_len:
                x = x[:seq_max_len, :]
            x = np.nan_to_num(x)
            dataset['seq_len'].append(len(x))
            x = np.pad(x, ((0, seq_max_len - len(x)), (0, 0)), 'constant', constant_values=0)
            dataset['test'].append(x)

    dataset['test'] = np.array(dataset['test'])
    dataset['id'] = np.array(dataset['id'])
    dataset['seq_len'] = np.expand_dims(np.array(dataset['seq_len']), axis=1)
    print(dataset['test'].shape, dataset['seq_len'].shape, dataset['id'].shape)
    with open("testing_data.pkl", "wb") as pkl_f:
        pickle.dump(dataset, pkl_f, protocol=4)


def classification_evaluation(y_ture, y_pred):
    acc = accuracy_score(y_ture, (y_pred > 0.5).astype('int'))
    auc = roc_auc_score(y_ture, y_pred)
    fpr, tpr, thresholds = roc_curve(y_ture, y_pred)
    return acc, auc
    # print('Accuracy:', acc)
    # print('ROC AUC Score:', auc)


def batch_generator_new(train_dict, batch_size=512):
    ones_ids = np.squeeze(np.where(train_dict['labels'][:, 0] == 1))
    zeors_ids = np.squeeze(np.where(train_dict['labels'][:, 0] == 0))

    print('ones to zeros percentage: ', len(zeors_ids) / len(ones_ids), 'number of ones: ', len(ones_ids),
          'number of zeros: ', len(zeors_ids))
    while True:
        zeors_ids_selected = np.random.choice(zeors_ids, batch_size // 2, replace=False)
        ones_ids_selected = np.random.choice(ones_ids, batch_size - len(zeors_ids_selected), replace=False)
        selected_ids = np.concatenate([zeors_ids_selected, ones_ids_selected], axis=0)
        np.random.shuffle(selected_ids)

        x_batch = train_dict['train'][selected_ids]
        y_batch = train_dict['labels'][selected_ids]
        seqlen_batch = train_dict['seq_len'][selected_ids]

        yield (np.squeeze(x_batch), y_batch, np.squeeze(seqlen_batch))


def get_RNNCell(cell_types, keep_prob, state_size, build_with_dropout=True):
    """
    Helper function to get a different types of RNN cells with or without dropout wrapper
    :param cell_types: cell_type can be 'GRU' or 'LSTM' or 'LSTM_LN' or 'GLSTMCell' or 'LSTM_BF' or 'None'
    :param keep_prob: dropout keeping probability
    :param state_size: number of cells in a layer
    :param build_with_dropout: to enable the dropout for rnn layers
    :return:
    """
    cells = []
    for cell_type in cell_types:
        if cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=state_size,
                                          bias_initializer=tf.zeros_initializer())  # Or GRU(num_units)
        elif cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(num_units=state_size, use_peepholes=True, state_is_tuple=True,
                                           initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_LN':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(state_size)
        elif cell_type == 'GLSTMCell':
            cell = tf.contrib.rnn.GLSTMCell(num_units=state_size, initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_BF':
            cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=state_size, use_peephole=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

        if build_with_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    if build_with_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell

def conv_layer(X,filters,kernel_size,strides,padding):
    """
    1D convolutional layer with or without dropout or batch normalization
    :param batch_norm:  bool, enable batch normalization
    :param is_train: bool, mention if current phase is training phase
    :param scope: variable scope
    :return: 1D-convolutional layer
    """
    return tf.layers.conv1d(inputs=X, filters=filters, kernel_size=kernel_size, strides=strides,
                                     padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     activation=tf.nn.relu)

def dense_layer(x, size,activation_fn, drop_out=False, keep_prob=None):
    """
    Helper function to create a fully connected layer with or without batch normalization or dropout regularization
    :param x: previous layer
    :param size: fully connected layer size
    :param activation_fn: activation function
    :param batch_norm: bool to set batch normalization
    :param phase: if batch normalization is set, then phase variable is to mention the 'training' and 'testing' phases
    :param drop_out: bool to set drop-out regularization
    :param keep_prob: if drop-out is set, then to mention the keep probability of dropout
    :param scope: variable scope name
    :return: fully connected layer
    """
    return_layer = tf.layers.dense(x, size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=activation_fn)
    if drop_out:
        return_layer = tf.nn.dropout(return_layer, keep_prob)

    return return_layer

def split_data(train_dict, split=512):
    ones_ids = np.squeeze(np.where(train_dict['labels'][:, 0] == 1))
    zeors_ids = np.squeeze(np.where(train_dict['labels'][:, 0] == 0))
    print('ones to zeros percentage: ', len(zeors_ids) / len(ones_ids), 'number of ones: ', len(ones_ids),
          'number of zeros: ', len(zeors_ids))

    ones_x = train_dict['train'][ones_ids]
    ones_y = train_dict['labels'][ones_ids]
    ones_seq = train_dict['seq_len'][ones_ids]

    zeors_x = train_dict['train'][zeors_ids]
    zeors_y = train_dict['labels'][zeors_ids]
    zeors_seq = train_dict['seq_len'][zeors_ids]

    ones_x_train, ones_x_vali, ones_y_train, ones_y_vali, ones_seq_train, ones_seq_vali = train_test_split(ones_x, ones_y, ones_seq, test_size=split, random_state=5228)

    zeors_x_train, zeors_x_vali, zeors_y_train, zeors_y_vali, zeors_seq_train, zeors_seq_vali = train_test_split(zeors_x, zeors_y, zeors_seq, test_size=split, random_state=5228)

    x_train = np.concatenate([ones_x_train, zeors_x_train], axis=0)
    y_train = np.concatenate([ones_y_train, zeors_y_train], axis=0)
    seq_train = np.concatenate([ones_seq_train, zeors_seq_train], axis=0)

    x_vali = np.concatenate([ones_x_vali, zeors_x_vali], axis=0)
    y_vali = np.concatenate([ones_y_vali, zeors_y_vali], axis=0)
    seq_vali = np.concatenate([ones_seq_vali, zeors_seq_vali], axis=0)

    print(x_train.shape, y_train.shape, seq_train.shape, x_vali.shape, y_vali.shape, seq_vali.shape)

    return {'train': x_train, 'labels': y_train, 'seq_len': seq_train}, {'test': x_vali, 'labels': y_vali, 'seq_len': seq_vali}


def model():
    train_dict, test_dict = data_generation(load=True)
    x_test, id_test, seq_test = test_dict['test'], test_dict['id'], test_dict['seq_len']

    print("all training data: ", train_dict['train'].shape, "all testing data: ", test_dict['test'].shape)

    train_dict, vali_dict = split_data(train_dict)
    x_vali, y_vali, seq_vali = vali_dict['test'], vali_dict['labels'], vali_dict['seq_len']



    seq_max_len = 70
    input_dim = train_dict['train'].shape[-1]
    rnn_layers = 2
    rnn_hidden_units = 100
    batch_size = 512
    epoches = 10
    # iternations = x_train.shape[0] // batch_size + 1
    iternations = 4000

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_dict['train'].reshape(-1, input_dim))
    train_shape = train_dict['train'].shape
    vali_shape = x_vali.shape
    test_shape = x_test.shape
    train_dict['train'] = scaler.transform(train_dict['train'].reshape(-1, train_shape[-1])).reshape(train_shape)
    x_vali = scaler.transform(x_vali.reshape(-1, vali_shape[-1])).reshape(vali_shape)
    x_test = scaler.transform(x_test.reshape(-1, test_shape[-1])).reshape(test_shape)

    MODE = 'TEST'

    x = tf.placeholder(tf.float32, [None, seq_max_len, input_dim])
    seq_len = tf.placeholder(tf.int32, [None])
    y = tf.placeholder(tf.int32, [None, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    cell = get_RNNCell(['LSTM'] * rnn_layers, keep_prob=keep_prob, state_size=rnn_hidden_units)
    outputs, states = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_len, dtype=tf.float32)

    # cell_fw = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_units, use_peepholes=True, state_is_tuple=True,
    #                                initializer=tf.contrib.layers.xavier_initializer())
    # cell_bw = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_units, use_peepholes=True, state_is_tuple=True,
    #                                   initializer=tf.contrib.layers.xavier_initializer())
    # outputs, _= tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length=seq_len, dtype=tf.float32)

    # outputs = conv_layer(outputs, filters=64, kernel_size=3, strides=1, padding='same')
    # outputs = tf.layers.max_pooling1d(inputs=outputs, pool_size=2, strides=2, padding='same', name='maxpool_1')

    # outputs = conv_layer(outputs, filters=32, kernel_size=3, strides=1, padding='same')
    # outputs = tf.layers.max_pooling1d(inputs=outputs, pool_size=2, strides=2, padding='same', name='maxpool_2')

    # outputs = conv_layer(outputs, filters=32, kernel_size=2, strides=1, padding='same')
    # outputs = tf.layers.max_pooling1d(inputs=outputs, pool_size=2, strides=2, padding='same', name='maxpool_3')
    #
    # output = tf.layers.flatten(outputs)

    # shape = outputs.get_shape().as_list()

    batch_size_tf = tf.shape(outputs)[0]

    index = tf.range(0, batch_size_tf) * seq_max_len + (seq_len - 1)

    output = tf.gather(tf.reshape(outputs, [-1, rnn_hidden_units]), index)

    logits = tf.layers.dense(output, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=None)


    # y_onehot = tf.squeeze(tf.one_hot(y, 2))
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_onehot))

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(y, tf.float32)))

    global_step = tf.Variable(0, trainable=False)
    learning_rate = 0.001
    decay_steps = 200
    decay_rate = 0.1
    learning_rate = tf.compat.v1.train.inverse_time_decay(learning_rate,
                                                          global_step,
                                                          decay_steps, decay_rate, staircase=True)

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step)

    # correct_pred = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y_onehot, 1))
    # acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # prediction = tf.argmax(tf.nn.softmax(logits), 1)
    prediction = tf.nn.sigmoid(logits)

    # training_batch = batch_generator(x_train, y_train, seq_len_train, batch_size=batch_size)
    # testing_batch = batch_generator(x_test, y_test, seq_len_test, batch_size=batch_size)
    new_batch = batch_generator_new(train_dict)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    from sklearn.metrics import accuracy_score
    with tf.Session() as sess:
        sess.run(init)
        if MODE == 'TRAIN':
            for j in range(iternations):
                x_batch, y_batch, seqlen_batch = next(new_batch)
                # print(x_batch.shape, y_batch.shape, seqlen_batch.shape)
                feed_dict = {x: x_batch, y: y_batch, seq_len: np.squeeze(seqlen_batch), keep_prob: 0.6}
                # print(sess.run(outputs, feed_dict=feed_dict).shape)
                sess.run(train_op, feed_dict=feed_dict)


                feed_dict_train = {x: x_batch, y: y_batch, seq_len: np.squeeze(seqlen_batch), keep_prob: 1.0}
                a, train_c = sess.run([prediction, cost], feed_dict=feed_dict_train)
                train_acc , train_auc = classification_evaluation(y_batch, a)

                feed_dict_test = {x: x_vali, y: y_vali, seq_len: np.squeeze(seq_vali), keep_prob: 1.0}
                a, test_c = sess.run([prediction, cost], feed_dict=feed_dict_test)
                test_acc, test_auc = classification_evaluation(y_vali, a)

                print(sess.run(global_step), ': train: ', train_acc , train_auc , train_c, "\ttest: ", test_acc, test_auc, test_c)
                if j % 100 == 0:
                    save_path = saver.save(sess, './model_checkpoints/')
        elif MODE == 'TEST':
            saver.restore(sess, './model_checkpoints/')

            solution = np.zeros((len(id_test), 2))

            feed_dict = {x: x_test, seq_len: np.squeeze(seq_test), keep_prob: 1.0}
            pred_vali = sess.run(prediction, feed_dict=feed_dict)
            print("test prediction: ", pred_vali.shape)

            import pandas as pd
            # pd.set_option('display.float_format', lambda x: '%.15f' % x)
            for i in tqdm(range(len(id_test))):
                index_id = id_test[i]
                solution[index_id][0] = index_id
                solution[index_id][1] = pred_vali[i]

            data = {'Id': solution[:, 0].astype(int), 'Predicted': solution[:, 1]}
            df = pd.DataFrame(data)
            df.to_csv('solution.csv', index=None, header=True, float_format='%.15f')

            print(solution.shape)


if __name__ == '__main__':
    model()

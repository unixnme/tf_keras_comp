import keras
import numpy as np
import tensorflow as tf


img_rows = 28
img_cols = 28
img_channels = 1
num_train_samples = 60000
num_test_samples = 10000
num_classes = 10
trainig_epochs = 10
batch_size = 128
test_batch_size = 1000
learning_rate = 0.001

class DataType(object):
    TRAIN = 'train'
    TEST = 'test'

def get_batches(batch_size=128, sparse=True, shuffle=True, datatype=DataType.TRAIN):
    """
    generator for getting batches of X,Y data for mnist
    :param batch_size: # of samples per batch
    :param sparse: if true, Y has shape (batch_size, 1)
                   if false, Y has shape (batch_size, num_classes)
    :param shuffle: if true, shuffle
                    if false: do not shuffle
    :param datatype: DataType object specifing whether train or test set
    :return: (X,Y) where X is normalized to float32 of [0,1]
            X.shape = (batch_size, 28, 28, 1)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x, y type in uint8
    # x.shape = (sample_size, 28, 28)
    # y.shape = (sample_size, )
    # sample_size = 60000 for train and 10000 for test
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = np.expand_dims(y_train, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    if datatype == DataType.TRAIN:
        num_samples = num_train_samples
    else:
        num_samples = num_test_samples

    while True:
        # never ending loop for the generator
        indices = np.arange(num_samples)
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            keys = indices[start:end]

            if datatype == DataType.TRAIN:
                x, y = x_train[keys], y_train[keys]
            else:
                x, y = x_test[keys], y_test[keys]

            if sparse is False:
                y = keras.utils.to_categorical(y, num_classes).astype(np.float32)
            else:
                y = y.astype(np.int32)

            yield x, y

def unit_test_get_batches():
    train_gen = get_batches()
    test_gen = get_batches(batch_size=15, sparse=False, shuffle=False, datatype=DataType.TEST)

    for __ in range(10):
        for _ in range(int(num_train_samples / float(128))):
            x,y = next(train_gen)
            assert x.shape == (128, img_rows, img_cols, 1) and y.shape == (128, 1)
        x,y = next(train_gen)
        assert x.shape == (num_train_samples % 128, img_rows, img_cols, 1) and y.shape == (num_train_samples % 128, 1)
        assert x.dtype == np.float32 and y.dtype == np.int32

        for _ in range(int(num_test_samples / float(15))):
            x,y = next(test_gen)
            assert x.shape == (15, img_rows, img_cols, 1) and y.shape == (15, num_classes)
        x,y = next(test_gen)
        assert x.shape == (num_test_samples % 15, img_rows, img_cols, 1) and y.shape == (num_test_samples % 15, num_classes)
        assert x.dtype == np.float32 and y.dtype == np.float32

def create_tf_model(depths=[32, 64], sparse=True):
    """
    create tensorflow model
    :param depths: list of convolution channels to append
    :param sparse: whether to use sparse categorical entropy
    :return: list of sequential tensors
    """
    x = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channels])
    if sparse:
        y = tf.placeholder(tf.int32, [None])
    else:
        y = tf.placeholder(tf.float32, [None, num_classes])

    W = []; b = []
    shape = [img_rows, img_cols, img_channels, None]
    for depth in depths:
        shape[-1] = depth
        W.append(tf.Variable(tf.random_normal(shape)))
        b.append(tf.Variable(tf.random_normal([depth])))
        shape[0] /= 2; shape[1] /= 2; shape[2] = depth

    W.append(tf.Variable(tf.random_normal([shape[0] * shape[1] * depths[-1], num_classes])))
    b.append(tf.Variable(tf.random_normal([num_classes])))

    tensors = [y, x]
    for idx in range(len(depths)):
        tensors.append(tf.nn.conv2d(tensors[-1], filter=W[idx], strides=[1,1,1,1], padding='SAME') + b[idx])
        tensors.append(tf.nn.elu(tensors[-1]))
        tensors.append(tf.nn.max_pool(tensors[-1], ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'))

    tensors.append(tf.reshape(tensors[-1], [-1, shape[0]*shape[1]*depths[-1]]))
    tensors.append(tf.matmul(tensors[-1], W[-1]) + b[-1])

    if sparse:
        tensors.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tensors[-1], labels=y))
    else:
        tensors.append(tf.nn.softmax_cross_entropy_with_logits(logits=tensors[-1], labels=y))
    tensors.append(tf.reduce_mean(tensors[-1]))

    pred = tf.argmax(tensors[-3], axis=-1)
    return W, b, tensors, pred


if __name__ == '__main__':
    #unit_test_get_batches()
    W, b, tensors, pred = create_tf_model()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tensors[-1])

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # training
        gen = get_batches(batch_size=batch_size, sparse=True)
        gen_test = get_batches(batch_size=test_batch_size, sparse=True, datatype=DataType.TEST, shuffle=True)

        for epoch in range(trainig_epochs):
            avg_cost = 0.
            batches_per_epoch = int(np.ceil(num_train_samples / float(batch_size)))
            for i in range(batches_per_epoch):
                x,y = next(gen)
                f = {tensors[1]: x, tensors[0]: y.reshape(-1)}
                _, cost = sess.run([optimizer, tensors[-1]], f)
                avg_cost += cost
                #print 'batch', i, 'cost =', cost

            avg_cost /= batches_per_epoch
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

            x,y = next(gen_test)
            f = {tensors[1]: x, tensors[0]: y.reshape(-1)}
            y_pred, cost = sess.run([pred, tensors[-1]], f)
            accuracy = np.sum(y_pred == y.reshape(-1)) / float(test_batch_size)
            print 'test accuracy =', accuracy, '%'
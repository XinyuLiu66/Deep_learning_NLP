import tensorflow as tf
#import progressbar as pb
import numpy as np
import csv
import time
data_dim_with_bias=101
class Model:
    def __init__(self, learning_rate):
        # set up the model
        self.X = tf.placeholder(tf.float16, shape=(None, data_dim_with_bias))
        self.y = tf.placeholder(tf.float16, shape=(None, 1))
        self.w = tf.get_variable("weights",
                                 shape=(data_dim_with_bias, 1),
                                 dtype=tf.float16,
                                 initializer=tf.random_normal_initializer())
        self.perceptron = tf.sigmoid(tf.matmul(self.X, self.w))
        self.loss = tf.losses.mean_squared_error(self.y, self.perceptron)
        self.sgd = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


    def train_tf(self,sess, model, train_x, train_y, epochs, batch_size):
       # bar = pb.ProgressBar(max_value=epochs)
        for i in range(epochs):
            train_x_batches, train_y_batches = self.get_random_batches(train_x, train_y, batch_size)
            for X_batch, y_batch in zip(train_x_batches, train_y_batches):
                _, w_val = sess.run([model.sgd, model.w], feed_dict={model.X: X_batch, model.y: y_batch})

    def get_random_batches(self,X, y, batch_size):
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]
        # when using array_split for 100 datapoints and batch size 33 one would get batches [33, 33, 33, 1]
        X_batches = np.array_split(X, len(y)//batch_size)
        y_batches = np.array_split(y, len(y)//batch_size)
        return X_batches, y_batches

    def test_tf(self,sess, model, test_x, test_y):
        loss, predictions = sess.run([model.loss, model.perceptron], feed_dict={model.X: test_x, model.y: test_y})
        accuracy = np.sum(np.equal(test_y, np.rint(predictions))) / len(test_y)
        return loss, accuracy

    def reader(self, file):
        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter="\t")
            label = []
            code_label = []  # convert POS and NEG to 1 or 0
            input_data = []  # string type of review data
            inputData = []  # float type of review data

            # load label to list label, load review data to list input_data
            for row in readCSV:
                label.append(row[-2])
                input_data.append(row[-1])
            # convert POS and NEG to 1 and 0
            for w in label:
                if w.endswith('POS'):
                    code_label.append(1)
                elif w.endswith('NEG'):
                    code_label.append(0)
            # convert string type to float type
            for w in input_data:
                templist = []
                row_data = w.split(" ")
                for word in row_data:
                    data = float(word)
                    templist.append(data)
                inputData.append(templist)

            # append bias 1 to each x input
            for _ in inputData:
                _.append(1)
            # convert input label to numpy array
            np_input_label = np.array(code_label, dtype=np.int32)
            # convert input review data to numpy array 100 dimention
            np_input_data = np.array(inputData, dtype=np.float64)
            return np_input_label, np_input_data


    def run_tensorflow(self,epochs=100, batch_size=10, learning_rate=0.01):
        tf.set_random_seed(42)
        train_x, train_y = self.reader("rt-polarity.train.csv")
        test_x, test_y = self.reader("rt-polarity.test.csv")
        # set up the model
        model = Model(learning_rate)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.train_tf(sess, model, train_x, train_y, epochs, batch_size)

            # print results on dev and test
         #   loss, accuracy = self.test_tf(sess, model, dev_x, dev_y)
         #   print("Loss on dev after {} epochs: {}, accuracy: {}".format(epochs, loss, accuracy))
            loss, accuracy = self.test_tf(sess, model, test_x, test_y)
            print("Loss on test after {} epochs: {}, accuracy: {}".format(epochs, loss, accuracy))


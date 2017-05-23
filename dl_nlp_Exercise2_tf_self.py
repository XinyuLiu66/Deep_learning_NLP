import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Model:
    def __init__(self,learning_rate):
        self.X = tf.placeholder(tf.float16, shape=(None, 101))
        self.y = tf.placeholder(tf.float16, shape=None)
        self.w = tf.get_variable(name='weight',shape = (101,1),
                                dtype = tf.float16,
                                initializer = tf.random_normal_initializer())
        self.perceptron = tf.sigmoid(tf.matmul(self.X,self.w))
        self.loss = tf.losses.mean_squared_error(self.y,self.perceptron)
        self.sgd = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)


# ====================get_ random==================================#
def get_random(X,y,mini_batch) :
    permu = np.random.permutation(len(y))
    X = X[permu]
    y = y[permu]
    X_batches = np.array_split(X,len(y)/mini_batch)
    y_batches = np.array_split(y, len(y) / mini_batch)
    return X_batches,y_batches

#=================================reader================================#
def reader(src):
    with open(src) as file:
        line_data = []
        review_data = []
        y = []
        for line in file:
            line_data.append(line)
        for sample in line_data:
            y.append(sample.split("\t")[1])
            review_data.append(sample.split("\t")[2])
       # print(review_data[:2])
        return y, review_data
# 1, Transfer data which has been read to machine readable data, that is transform label ="POS" to
#    1, NEG to 0 .
# 2  append bias 1 to each review_data sample,to make one sample from 100 dimension to 101
#          =============================
def getInputData(y,review_data) :
    true_label = []
    reviewData = []
    for w in y:
        if(w[-3:] == "POS"):
            true_label.append(1)
        else:
            true_label.append(0)
    for sample in review_data:
        list_sample = []
        sample = sample.strip('\n')
        list_split_str =  sample.split(" ")
        for data in list_split_str:
            d = float(data)
            list_sample.append(d)
        list_sample.append(float(1.0))
        reviewData.append(list_sample)
    #print(reviewData[0],"  ",  len(reviewData[0]),'  ',reviewData[0][-3:],"  ",type(reviewData[0][-2]))
    # to numpy
    true_label = np.array(true_label)
    reviewData = np.array(reviewData)
    return true_label,reviewData
#====================train data====================================#
def train_tf(sess, model, train_x, train_y, epochs, batch_size):
    X_batches,y_batches = get_random(train_x,train_y,batch_size)
    x_index = []
    loss_train_list = []
    loss_dev_list = []
    for i in range(epochs):
        for X_batch,y_batch in zip(X_batches,y_batches):
            _,w = sess.run([model.sgd,model.w],feed_dict = {model.X:X_batch, model.y:y_batch})
        x_index.append(i)
        loss_train, _ = test_tf(sess, model, train_x, train_y)
        loss_dev, _ = test_tf(sess, model, dev_x, dev_y)
        loss_train_list.append(loss_train)
        loss_dev_list.append(loss_dev)
    plt.plot(x_index,loss_train_list,'r')
    plt.plot(x_index,loss_dev_list,'b')

#====================test data====================================#
def test_tf(sess, model, test_x, test_y):
    loss,prediciton= sess.run([model.loss,model.perceptron],
                              feed_dict = {model.X:test_x, model.y:test_x})
    acc = np.sum([i == j for i,j in zip(np.rint(prediciton),test_y)])/len(test_y)
    return loss, acc

#================run tensorflow=================================#
def run_tensorflow(epochs=100, batch_size=10, learning_rate=0.01):

    tf.set_random_seed(1)
    model = Model(learning_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_tf(sess, model, train_x, train_y, epochs, batch_size)
        loss, accuracy = test_tf(sess, model, dev_x, dev_y)
        print("Loss on dev after {} epochs: loss: {}, accuracy: {}".format(epochs, loss, accuracy))
        loss, accuracy = test_tf(sess, model, test_x, test_y)
        print("Loss on test after {} epochs: loss: {}, accuracy: {}".format(epochs, loss, accuracy))


#====================================main()=======================#
src_train = "rt-polarity.train.vecs"
src_dev = "rt-polarity.dev.vecs"
src_test = "rt-polarity.test.vecs"
old_train_y,old_train_review_data = reader(src_train)
old_dev_y,old_dev_review_data = reader(src_dev)
old_test_y,old_test_review_data = reader(src_test)
train_y,train_x = getInputData(old_train_y,old_train_review_data)
dev_y,dev_x = getInputData(old_dev_y,old_dev_review_data)
test_y,test_x = getInputData(old_test_y,old_test_review_data)
fig = plt.figure()
run_tensorflow(epochs=100, batch_size=20, learning_rate=0.01)
plt.show()
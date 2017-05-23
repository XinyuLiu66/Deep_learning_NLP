import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
class Model:
    def __init__(self,learning_rate):
        self.X = tf.placeholder(dtype=tf.float32,shape=(None,101))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None))
        self.input_layer = add_layer(self.X, 101, 10, activation_function=tf.nn.tanh)
        self.hidden_layer1 = add_layer(self.input_layer,10,10,activation_function=tf.nn.tanh)
        self.hidden_layer2 = add_layer(self.hidden_layer1, 10, 10, activation_function=tf.nn.tanh)
        self.output        = add_layer(self.hidden_layer2, 10, 1, activation_function=tf.nn.tanh)  #None : the same rows with X
        self.loss = tf.losses.mean_squared_error(self.output,self.y)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

#====================================add hidden layer==========================================================
def add_layer(inputs,input_size,output_size,activation_function = None):
    with tf.name_scope("weight"):

        W = tf.Variable(tf.random_normal([input_size,output_size]),name="weights")

    #define pre_active
    pre_activate = tf.matmul(inputs,W)
    # define output  (input have been give)   --->
    # add a hidden layer, only need 3 things, input, preactive(Weight), output(active_function)
    if(activation_function == None):
        output = pre_activate
    else:
        output = activation_function(pre_activate)
    return output

#====================================get random data==========================================================
def get_random(training_X, training_Y, batch_size):
    tf.set_random_seed(1)
    per = np.random.permutation(len(training_Y))
    training_X = training_X[per]
    training_Y = training_Y[per]
    training_X_batches = np.array_split(training_X,len(training_Y)//batch_size)
    training_Y_batches = np.array_split(training_Y,len(training_Y)//batch_size)
    return training_X_batches,training_Y_batches

#====================================training data==========================================================
def training_data(sess,model,training_X,training_Y,epochs, batch_size):
    training_X_batches, training_Y_batches = get_random(training_X, training_Y, batch_size)
    x_index = []
    train_loss = []
    dev_loss = []
    fig = plt.figure()
    for step in range(epochs):
        for X_batch,Y_batch in zip(training_X_batches, training_Y_batches):
            sess.run([model.optimizer],feed_dict ={model.X: X_batch, model.y: Y_batch})
        print(test_data(sess, model, training_X, training_Y))
        x_index.append(step)
        train_loss.append(test_data(sess, model, training_X, training_Y))
        dev_loss.append(test_data(sess, model, dev_x, dev_y))
    plt.plot(x_index,train_loss,'r')
    plt.plot(x_index, dev_loss, 'b')
    plt.show()

# ====================================test data=============================================================
def test_data(sess,model,test_X,test_Y):
    test_loss = sess.run([model.loss],feed_dict ={model.X: test_X, model.y: test_Y})
    return test_loss


# ====================================run tensorflow=============================================================
def run_tf(learning_rate,epochs,batch_size):
    model = Model(learning_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_data(sess, model, train_x, train_y, epochs, batch_size)
        test_data(sess, model, test_x, test_y)

#=================================reader=======================================================================#
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

src_train = "rt-polarity.train.vecs"
src_dev = "rt-polarity.dev.vecs"
src_test = "rt-polarity.test.vecs"
old_train_y,old_train_review_data = reader(src_train)
old_dev_y,old_dev_review_data = reader(src_dev)
old_test_y,old_test_review_data = reader(src_test)
train_y,train_x = getInputData(old_train_y,old_train_review_data)
dev_y,dev_x = getInputData(old_dev_y,old_dev_review_data)
test_y,test_x = getInputData(old_test_y,old_test_review_data)

run_tf(learning_rate=0.01,epochs=100,batch_size=10)
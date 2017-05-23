import tensorflow as tf
#import progressbar as pb
import numpy as np
import matplotlib.pyplot as plt
import csv

figure = plt.figure()
ax =figure.add_subplot(1,1,1)
class Model:

    def __init__(self,learning_rate):
        self.x = tf.placeholder(tf.float32,shape=(None))
        self.y = tf.placeholder(tf.float32, shape=(None))
        self.w = tf.get_variable("weight", shape=(1),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer)
        self.bias = tf.get_variable("bias",shape=(1),
                                    dtype=tf.float32,
                                    initializer=tf.random_normal_initializer)
        self.perceptron = self.w * tf.square(self.x) + self.bias
        self.loss = tf.losses.mean_squared_error(self.perceptron,
                                                 self.y)
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        ).minimize(self.loss)

def traing_tf(sess,model,train_x,train_y,epoch):
    x_index = []
    y_index = []
    for i in range(epoch):
        _, w = sess.run([model.optimizer,model.w],feed_dict = {model.x:train_x,model.y:train_y})
        loss = test_tf(sess,model,train_x,train_y)
        predict = sess.run(model.perceptron,feed_dict = {model.x:train_x})
       # lines = ax.plot(train_x,predict,'r')
        print(loss)
        #plt.pause(0.1)
        #ax.lines.remove(lines[0])



def test_tf(sess,model,test_x,test_y):
    loss = sess.run([model.loss],feed_dict = {model.x:test_x,model.y:test_y})
    return loss

def run_tensor():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        traing_tf(sess, model, train_x, train_y, epoch=1000)

#===============main==============
model = Model(0.1)
ax = figure.add_subplot(111)
train_x = np.linspace(-1,1,300)[:,np.newaxis]
train_y = np.square(train_x)+1
plt.scatter(train_x,train_y)
run_tensor()
plt.show()
    # def run_session(self):
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         x_data = np.linspace(-1,1,300)
    #         y_data = np.square(x_data)+0.3
    #         ax.scatter(x_data,y_data)
    #         plt.title("training a line")
    #         plt.xlabel("x")
    #         plt.ylabel("y")
    #         plt.ion()
    #         plt.show()
    #         for step in range(200):
    #             sess.run(self.optimizer, feed_dict={self.x:
    #                                                     x_data,
    #                                                 self.y: y_data})
    #             prediction = sess.run(self.perceptron,feed_dict={self.x:
    #                                                     x_data,
    #                                                 self.y: y_data})
    #             if(step%10 == 0):
    #                 line = ax.plot(x_data,prediction,'r')
    #                 plt.pause(0.2)
    #                 ax.lines.remove(line[0])
    #                 print("w = ", sess.run(self.w))
    #                 print("bias = ", sess.run(self.bias))

model = Model(0.1)
model.run_session()

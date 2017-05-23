import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Activation

import numpy as np

#============================   reader     read data   ==========================================#
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

    # to numpy
    true_label = np.array(true_label)
    reviewData = np.array(reviewData)
    return true_label,reviewData

src_train = "rt-polarity.train.vecs"
src_dev = "rt-polarity.dev.vecs"
src_test = "rt-polarity.test.vecs"
old_train_y, old_train_review_data = reader(src_train)
old_dev_y, old_dev_review_data = reader(src_dev)
old_test_y, old_test_review_data = reader(src_test)
train_y, train_x = getInputData(old_train_y, old_train_review_data)
dev_y, dev_x = getInputData(old_dev_y, old_dev_review_data)
test_y, test_x = getInputData(old_test_y, old_test_review_data)


#====================================================================================================#


#============================   creat model   ======================================================#

model = Sequential()
#model.add(Dense(output_dim = 1, input_dim=1))
model.add(Dense(units = 10,input_dim=101,activation='tanh'))  #input layer
model.add(Dense(units = 10,activation='tanh'))    # hidden layer 1
model.add(Dense(units = 10,activation='tanh'))     # hidden layer 2
model.add(Dense(units = 1,activation='tanh'))      #output layer

model.compile(loss="mse",optimizer="sgd")

#====================================================================================================#


#=======================================get random================================================#
def get_random(train_x,train_y):
    tf.set_random_seed(1)
    per = np.random.permutation(len(train_y))
    x = train_x[per]
    y = train_y[per]
    return x,y


#===============================================train data============================================#

epochs = 100
x_index = []
y_index = []
#model.fit(train_x, train_y, epochs=100, batch_size=10)
for epoch in range(epochs):
    training_x,training_y = get_random(train_x,train_y)
    cost = model.train_on_batch(training_x,training_y)
    x_index.append(epoch)
    y_index.append(cost)
    print("train_cost = ", cost)
plt.plot(x_index,y_index)
plt.show()



#test data

cost = model.evaluate(test_x,test_y,batch_size=len(test_y))
print("test_cost = ",cost)
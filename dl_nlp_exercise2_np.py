import numpy as np

import matplotlib.pyplot as plt
# numpy reader

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

# loss/cost function   =============================
def loss_function(X, w, y):
    prediction = sigmond(np.dot(X,w))
    prediction_dict = [np.rint(i) for i in prediction]
    temp = np.square(sigmond(np.dot(X,w)) - y)
    loss = np.sum(temp)/len(y)
    accuracy = np.sum([i == j for i,j in zip(prediction_dict,y)])/len(y)
    return loss,accuracy

# sigma function   =============================
def sigmond(X):
    result = 1.0/(1.0 + np.exp(-X))
    result  = np.array(result)
    return result

# upadte weight =============================
def epoch(train_x, train_y, w_init, batch_size, learning_rate):
    w = w_init
    train_x_batches, train_y_batches = getRandom(train_x, train_y, batch_size)
    for X_batch, y_batch in zip(train_x_batches, train_y_batches):
        sigxw = sigmond(np.dot(X_batch, w))
        temp = sigxw - y_batch
        sig_deri_x_mul_w = sigxw * (1 - sigxw)
        temp2 = temp * sig_deri_x_mul_w * np.transpose(X_batch)
        w = w - (learning_rate/mini_batch)*np.sum(temp2 , axis=1)
    return w

##=================================================================
# def update_weight2(X,y,w,learning_rate,mini_batch):
#      temp = (sigmond(np.dot(X,w)) - y)
#      sig_deri_x_mul_w = sigmond(np.dot(X, w)) * (1 - sigmond(np.dot(X, w)))
#      temp2 = temp * sig_deri_x_mul_w * np.transpose(X)
#      new_w = w - (learning_rate/mini_batch)*np.sum(temp2 , axis=1)
#      return new_w
#===========random input===========
def getRandom(X,y,mini_batch) :
    permu = np.random.permutation(len(y))
    X = X[permu]
    y = y[permu]
    X_batches = np.array_split(X,len(y)/mini_batch)
    y_batches = np.array_split(y, len(y) / mini_batch)
    return X_batches,y_batches
# train data ===============================
src_train = "rt-polarity.train.vecs"
src_dev = "rt-polarity.dev.vecs"
src_test = "rt-polarity.test.vecs"
old_train_y,old_train_review_data = reader(src_train)
old_dev_y,old_dev_review_data = reader(src_dev)
old_test_y,old_test_review_data = reader(src_test)
train_y,train_x = getInputData(old_train_y,old_train_review_data)
dev_y,dev_x = getInputData(old_dev_y,old_dev_review_data)
test_y,test_x = getInputData(old_test_y,old_test_review_data)
w = np.random.normal(0,1,(101,))
learning_rate = 0.001
mini_batch = 1
epochs = 100

X_batches,y_batches = getRandom(train_x,train_y,mini_batch)

fig = plt.figure()
ax = fig.add_subplot(111)
x_axis = []
y_axis =[]
# for step in range(epochs):
#     for X_batch1,y_batch1 in zip(X_batches,y_batches):
#         w = update_weight1(X_batch1,y_batch1,w,learning_rate,mini_batch)
#     loss = loss_function(X_batch1,w,y_batch1)
#     x_axis.append(step)
#     y_axis.append(loss)

#     print(loss,'\n')
loss_train_list = []
loss_dev_list = []
for i in range(epochs):
    w = epoch(train_x, train_y, w, mini_batch, learning_rate)
    loss_train,_ = loss_function(train_x, w, train_y)
    loss_dev,_ = loss_function(dev_x, w, dev_y)
    x_axis.append(i)
    loss_train_list.append(loss_train)
    loss_dev_list.append(loss_dev)
_,dev_acc = loss_function(dev_x, w, dev_y)
_,test_acc = loss_function(test_x, w, test_y)
print("dev_acc = ",dev_acc,"test_acc = ",test_acc)
plt.plot(x_axis, loss_train_list,'r')
plt.plot(x_axis, loss_dev_list,'b')
plt.show()
#print(data[1])

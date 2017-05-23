import numpy as np
import tensorflow as tf
import csv
import math
import random
import matplotlib.pyplot as plt
#========read file data=======#
def reader(file):
    with open(file) as csvfile :
        readCSV = csv.reader(csvfile , delimiter ="\t")
        label = []
        code_label = []   # convert POS and NEG to 1 or 0
        input_data = []   # string type of review data
        inputData = []    # float type of review data

        #load label to list label, load review data to list input_data
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
        np_input_label = np.array(code_label,dtype=np.int32)
        # convert input review data to numpy array 100 dimention
        np_input_data = np.array(inputData,dtype=np.float64)
        return np_input_label, np_input_data

#============================================#
# define sigmond activation function

def sigmond(x_array):
    try:
        result = [1/(1 + math.exp(-x)) for x in x_array]
    except OverflowError:
        result = float('inf')
    result = np.array(result)
    return result
#============================================#
#define function for weight update and return loss
def weight_update_and_loss(w,np_input_label,np_input_data,mini_batch,alpha):

    #shuffle the input data
    input = list(zip(np_input_label,np_input_data))
    random.shuffle(input)

    #seperate input review data and label
    input_review_data = [a[1] for a in input]
    input_label_class = [a[0] for a in input]

    np.random.seed(0)
   # mini_batch = np.random.randint(15)

    x = input_review_data[:mini_batch]
    y = input_label_class[:mini_batch]


    sig_x_mul_w = sigmond(np.dot(x,w))
    sig_minus_y = sig_x_mul_w - y

    sig_deri_x_mul_w = sigmond(np.dot(x,w)) * (1 - sigmond(np.dot(x,w)))
    temp1 = sig_minus_y * sig_deri_x_mul_w
    x_transpose = np.transpose(x)
    temp2 =temp1 * x_transpose
    sum_value = np.sum(temp2,axis = 0)

    # loss function
    loss = np.sum(np.square(sig_minus_y) )

    w_new = w - (alpha/mini_batch) * sum_value

    # calculate accuracy
    prediction = []
    for num in sig_minus_y:
        if(num>0.5 and num == 0.5):
            prediction.append(1)
        elif(num<0.5):
            prediction.append(0)

    accuracy = cal_accuracy(prediction, np_input_label)

    return w_new,loss, accuracy

#===========calculate the accuracy==============#
def cal_accuracy(a_array , b_array):
    size = len(a_array)
    counter = 0
    for i in range(size):
        if(a_array[i] == b_array[i]):
            counter += 1
    try:
        acc = counter/size
    except ZeroDivisionError:
        acc = 0
    return acc





input = reader("rt-polarity.train.csv")
input_label = input[0]
input_review_data = input[1]

# initial weight w, learning rate :alpha
w = np.random.normal(0,1,size=101)
alpha = 0.01
#np_input_label,np_input_data
figure = plt.figure()
ax  = figure.add_subplot(111)
loss_list = []
for step in range(300):
    w_loss = weight_update_and_loss(w,input_label,input_review_data,10,0.01)
    w = w_loss[0]
    loss = w_loss[1]
    loss_list.append(loss)
    accuracy = w_loss[2]

plt.title("Training Loss with learning rate = 0.01, batch size = 10")
plt.xlabel("Training Step")
plt.ylabel("Training Loss")
step_list = range(300)
ax.plot(step_list,loss_list)
plt.show()
  #  if(step % 10 == 0):
  #      print("loss",loss,'   acc',accuracy)

input = reader("rt-polarity.test.csv")
test_input_label = input[0]
test_input_review_data = input[1]
test_w_loss = weight_update_and_loss(w,input_label,input_review_data,50,0.01)
w = w_loss[0]
loss = w_loss[1]
accuracy = w_loss[2]
print("loss",loss,'   acc',accuracy)

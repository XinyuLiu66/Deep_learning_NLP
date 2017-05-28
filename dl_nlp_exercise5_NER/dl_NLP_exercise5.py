import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import preprocessing
import matplotlib.pyplot as plt

#begin=====================read file=======================================//
training_data_path ="/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise3_NE/hex05_data_v2/ner_eng_bio.train"
develop_data_path ="/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise3_NE/hex05_data_v2/ner_eng_bio.dev"
test_data_path ="/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise3_NE/hex05_data_v2/ner_eng_bio.test"

def reader(file_path):
    words_list = []
    sentences_list = []
    temp_word = []
    temp_word_property = []
    words_properties_list = []
    sentences_properties_list = []
    with open(file_path) as file:
        for line in file:
            l = line.split(" ")
            word = l[0]
            word_property = l[1:]
            if word != '\n':
                temp_word.append(word)
                temp_word_property.append(word_property)
            else:
                sentences_list.append(temp_word)
                sentences_properties_list.append(temp_word_property)
                temp_word = []
                temp_word_property = []
            words_list.append(word)
            words_properties_list.append(word_property)
    #print(len(words_list))
    #print(len(words_properties_list))

    return sentences_list, sentences_properties_list


#begin=====================data preprocessing=======================================//

def data_preprocessing(k,sentences_list, sentences_properties_list):
    sentences_NE_tag = []
    # remove two useless properties
    for snetences_prop in sentences_properties_list:
        temp_tag_list = []
        for word_prop in snetences_prop:
            temp_tag_list.append(word_prop[2])
        sentences_NE_tag.append(temp_tag_list)
    # print(sentences_NE_tag[:3])
    # print("np_tag_shape", np.shape(sentences_NE_tag))
    num_to_move = k//2
    padding_list = ["__PADDING__" for i in range(num_to_move)]

    new_sentence_list = []
    for sentence in sentences_list:
        new_sentence = []
        new_sentence.extend(padding_list)
        new_sentence.extend(sentence)
        new_sentence.extend(padding_list)
        new_sentence_list.append(new_sentence)

    total_window = []
    for sentence in new_sentence_list:
        sentence_window = []
        i = num_to_move
        while(i < len(sentence) - num_to_move):
            window = []
            left = i - num_to_move
            right = i + num_to_move
            m = left
            while(m <= right):
                window.append(sentence[m])
                m += 1
            sentence_window.append(window)
            i += 1
        total_window.append(sentence_window)
    # print(len(total_window))       #################################
    # print(len(sentences_NE_tag))

    return total_window, sentences_NE_tag


# sentences_list, sentences_properties_list = reader(training_data_path)
# total_window,sentences_NE_tag = data_preprocessing(3,sentences_list, sentences_properties_list)
# print(total_window[:3])
#print(sentences_NE_tag[:3])

#============================evaluation method using accuracy==================================

def evaluation_accuray(predict_list, golden_label_list):
    count = 0
    for i in range(len(predict_list)):
        if predict_list[i] == golden_label_list[i]:
            count += 1
    accuracy  = count/len(predict_list)
    return accuracy

def evaluation_f_meature(predict_list, golden_label_list):
    tp = 0
    true_label = 0
    predict_true_label = 0
    for i in range(len(golden_label_list)):
        if golden_label_list[i] != 'O\n'and predict_list[i] == golden_label_list[i]:
            tp += 1
        if golden_label_list[i] != 'O\n':
            true_label += 1
        if predict_list[i] != "O":
            predict_true_label += 1
    precision = tp/predict_true_label
    recall = tp/true_label
    f = 2 * precision *recall/(precision + recall)
    return f

#=============================encoding output ===========================================#

#encoding output,that is 'B-ORG\n', 'I-MISC\n', 'B-PER\n', 'I-PER\n', 'B-LOC\n', and so on with one hot vector

sentences_list, sentences_properties_list = reader(training_data_path)
tag = []
for sentence in sentences_properties_list:
    for w in sentence:
        tag.append(w[2])
np.random.seed(0)
output = list(set(tag))
output.sort()   #set disorder , so we sort the sequence

#!!!!!!!!!!importent!!!!!!!!!!
encoder = preprocessing.LabelEncoder()
encoder.fit(output)
encoder_OUTPUT = encoder.transform(output)
encode_label = np_utils.to_categorical(encoder_OUTPUT)
# print(output)
# print(encoder_OUTPUT)
# print(encode_label)
output_dict = {}
for i in range(len(output)):
    output_dict[output[i]] = encode_label[i]
#print(output_dict)


#==========================read word vector embeding=======================#

def read_word_Embedding(file_path):
    with open(file_path) as file:
        #words_list = []
        words_dict={}
        for line in file:
            l = line.split(" ")
            word_vetor = [float(w) for w in l[1:]]
            words_dict[l[0]] = word_vetor
           # words_list.append(l)
    return words_dict

word_vector_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise3_NE/hex05_embeddings/hex05.glove.6B.50d.txt"
words_dict = read_word_Embedding(word_vector_path)
#print(words_dict["for"])
#print(len(words_dict["for"]))


#==================================using keras to construct neuro network==============================#

#============================   creat model   ======================================================#

model = Sequential()
model.add(Dense(units=10,input_dim=50,activation="softmax"))   #input layer
model.add(Dense(units=10,activation="softmax"))    #hidden layer 1
model.add(Dense(units=9,activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="sgd")




#=========================================================================================#

#transform all the window and tag based on the each sentence to all the sample (62404, 50)

def transform_window_form(total_window):
    input_window = []
    for sentence_window in total_window:
        for word_window in sentence_window:
            input_window.append(word_window)


    input_vector = []
    for word_window in input_window:
        vector_for_word_window = np.zeros(50)
        for word in word_window:
            if words_dict.__contains__(word):
                vector_for_word = words_dict[word]
            elif word == "__PADDING__":
                np.random.seed(0)
                vector_for_word = np.random.normal(0,1,(50,))
            else:
                vector_for_word = np.zeros(50,)

            vector_for_word_window += vector_for_word
        vector_for_word_window = vector_for_word_window / k
        input_vector.append(vector_for_word_window)

    input_vector = np.array(input_vector)
    #!!!!!!!!!!in order to fit the input type of keras,
    #!!!!!!!!! must transform simple list of python to numpy array
    return input_vector







#===============================================================================================#
#transform all the output tag based on the each sentence to all the sample (62404, 50)


def transform_tag_form(sentences_NE_tag):


    output_y = []
    for sentence_tag in sentences_NE_tag:
        for word_tag in sentence_tag:
            word_vector = output_dict[word_tag]
            output_y.append(word_vector)
    output_y = np.array(output_y)
    #!!!!!!!!!!in order to fit the input type of keras,
    #!!!!!!!!! must transform simple list of python to numpy array
    return output_y




#=========================================train data======================================================#
k = 3
sentences_list, sentences_properties_list = reader(training_data_path)
total_window,sentences_NE_tag = data_preprocessing(k,sentences_list, sentences_properties_list)

x_index = []   # in order to show cost

y_index = []

input_vector = transform_window_form(total_window)
output_y = transform_tag_form(sentences_NE_tag)
for epoch in range(100):
   cost = model.train_on_batch(input_vector , output_y)
   x_index.append(epoch)
   y_index.append(cost)
   print(cost)
plt.plot(x_index,y_index)
#plt.show()
#！！！！！！！！！the below is the method which train a single sample each time, the cost not convergence,
#!!!!!!!!!!! so adapt to the upper method, train 100 epochs, each time train 62404 samples


#=========================================test data======================================================#

#get test data
test_sentences_list, test_sentences_properties_list = reader(test_data_path)
test_total_window,test_sentences_NE_tag = data_preprocessing(k,test_sentences_list, test_sentences_properties_list)
test_input_vector = transform_window_form(test_total_window)
test_output_y = transform_tag_form(test_sentences_NE_tag)

#test data
test_cost = model.evaluate(test_input_vector,test_output_y,batch_size=len(test_output_y))
print("test cost", test_cost)


#get accuracy

predict_output = model.predict(test_input_vector)
predict_output = (predict_output == predict_output.max(axis=1)[:,None]).astype(float) #!!!!!change max in each row to 1, all other numbers to 0


predict_tag = []
# decoding output from one-hot-vector to tag

output_decoding_dict = {}
for key,value in output_dict.items():
    output_decoding_dict[tuple(value)] = key

for p in predict_output:
     predict_tag.append(output_decoding_dict[tuple(p)])

accuracy = evaluation_accuray(predict_tag, test_output_y)
#f_meature = evaluation_f_meature(predict_tag, test_output_y)


print(accuracy)
#print(f_meature)



# for sentence_window,sentence_tag in zip(total_window,sentences_NE_tag):
#     for word_window,word_tag in zip(sentence_window,sentence_tag):
#         vector_for_word_window = np.zeros(50)
#         epoch += 1         # in order to show cost
#         x_index.append(epoch)     # in order to show cost
#         for word in word_window:
#             if words_dict.__contains__(word):
#                 vector_for_word = words_dict[word]
#             elif word == "__PADDING__":
#                 np.random.seed(0)
#                 vector_for_word = np.random.normal(0,1,(50,))
#             else:
#                 vector_for_word = np.zeros(50,)
#
#             vector_for_word_window += vector_for_word
#
#         vector_for_word_window = vector_for_word_window/k
#         vector_for_word_window = np.reshape(vector_for_word_window,newshape=(1,50))
#         #print(np.shape(vector_for_word_window))
#         y = output_dict[word_tag]
#         y = np.reshape(y,newshape=(1,9))
#         cost = model.train_on_batch(vector_for_word_window,y)
#         print(cost)
#         y_index.append(cost)
# plt.plot(x_index,y_index)
# plt.show()
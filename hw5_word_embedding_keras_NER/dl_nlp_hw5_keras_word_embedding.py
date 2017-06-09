import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from sklearn import preprocessing


#=====================reader======================#

def load_data_sets(path):
    sentences = []
    with open(path) as file:
        sentence = []
        for line in file:
            if(line.strip() == ""):
                sentences.append(sentence)
                sentence = []
            else:
                line = line.strip('\n')
                sentence.append(line.split(' '))
    return sentences

train_data_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise5_NER/hex05_data_v2/ner_eng_bio.train"
dev_data_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise5_NER/hex05_data_v2/ner_eng_bio.dev"
test_data_path = "/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise5_NER/hex05_data_v2/ner_eng_bio.test"

sentences_train = load_data_sets(train_data_path)
sentences_dev = load_data_sets(dev_data_path)
sentences_test = load_data_sets(test_data_path)




#====================data preprocessing==================
from itertools import tee
padding_token = "__PADDING__"
task_to_column = {"pos": 1, "chunk": 2, "ner": 3}


def apply_windowing(sentences, task, k):
    tag_column = task_to_column[task]

    windowed_sentences = []
    for sentence in sentences:
        # keep only the words and the tags for the given task
        filtered_sentence = [(tup[0], tup[tag_column]) for tup in sentence]

        # we need to pad the sentence beginning and sentence end with k//2 dummy words
        padding = [(padding_token, None)] * (k // 2)
        padded_sentence = padding + filtered_sentence + padding

        # create k iterators, then shift the iterators one element apart from each other
        iterators = tee(padded_sentence, k)
        for i, iterator in enumerate(iterators):
            for _ in range(i):
                next(iterator)

        # zip the iterators to obtain windows
        windowed_sentence = []
        for window in zip(*iterators):
            _, window_tag = window[k // 2]
            window_words = [word for word, tag in window]
            windowed_sentence.append((window_words, window_tag))
        windowed_sentences.append(windowed_sentence)

    return windowed_sentences

def flatten(windowed_sentences):
    """Flattens the given windowed sentences into a list of windows (sentence structure is not needed)."""
    return [tup for sentence in windowed_sentences for tup in sentence]

# test already pre processing data
    # train_data = apply_windowing(sentences_train, 'pos', 3)
    # train_data = flatten(train_data)
    # print(train_data[:3])




#===============evaluation ===================

from sklearn.metrics import f1_score
def accuracy(y_true, y_pred):
    return sum([1 for true, pred in zip(y_true,y_pred) if true==pred])/len(y_true)

def F1(y_true, y_pred, list_of_tags):
    return f1_score(y_true,y_pred,labels=list_of_tags,average="macro")



#=============Implement a multilayer perceptron with keras===================

np.random.seed(42)

from keras.models import Sequential
from keras.layers import Dense,Flatten,Embedding,Activation
from keras.optimizers import SGD
import re
def prepare_embedding_matrix(filename, embeddings_path="/Users/apple/Documents/tu_darmstadt/MSC_2semester/DL und NLP/uebung/dl_nlp_exercise5_NER/hex05_embeddings/"):
    # extract the embedding dimension from the filename (simpler solutions are imaginable)
    embedding_dimension = int(re.search(r"\d+(?=d)", filename).group(0))
    with open(embeddings_path + filename, 'r') as file:
        vocabulary_size = sum([1 for line in file])
        # reset the file pointer to the beginning of the file
        file.seek(0)   ###########
        vocabulary_size += 2
        embedding_matrix = np.zeros((vocabulary_size,embedding_dimension),dtype=np.float16)
        word_to_embedding_index = {}

        # index 0 for OOV words
        embedding_matrix[0] = np.random.randn(embedding_dimension)
        # # index 1 for __PADDING__
        embedding_matrix[1] = np.random.randn(embedding_dimension)

        word_to_embedding_index[padding_token] = 1
        # starting with index 2, we enter the regular words
        for i,line in enumerate(file,2):
            parts = line.split()
            word_to_embedding_index[parts[0]] = i
            embedding_matrix[i] = np.array(parts[1:],dtype=np.float16)
    return embedding_matrix, word_to_embedding_index

# test for embedding_matrix and word_to_embedding_index
embedding_filename = "hex05.glove.6B.50d.txt"
embedding_matrix, word_to_embedding_index = prepare_embedding_matrix(embedding_filename)
print('embedding matrix =',embedding_matrix[2])
print("word_to_embedding_index =",word_to_embedding_index['the'])


#=======build model========#

def build_model(k, embedding_matrix,list_of_tags):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=embedding_matrix.shape[1],
                        input_length=k,
                        weights=[embedding_matrix]))
    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('tanh'))
    model.add(Dense(units=len(list_of_tags),activation='tanh')) # as many outputs as there are tags (one-hot)
    model.add((Activation('softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')
    return model

def encode_data(windows,k,list_of_tags,embeedding_index):
    # use index to represent input x, because it will through Embedding layer
    x = []
    y = []
    for window,tag in windows:
        temp_x = []
        temp_y = [0] * len(list_of_tags)
        for w in window:
            temp_x.append(embeedding_index.get(w,0))
        x.append(temp_x)
        temp_y[list_of_tags.index(tag)] = 1
        y.append(temp_y)
    x = np.array(x)
    y = np.array(y)
    print('x =',x.shape)
    print('y = ',y.shape)
    return x, y

def decode_output(one_hot_prediction, list_of_tags):
    # determine the index of the highest value in each one-hot-vector, then look up the corresponding tag
    return [list_of_tags[i] for i in np.argmax(one_hot_prediction, axis=1)]

def run(task="pos", k=7, embedding_filename="hex05.glove.6B.50d.txt", eval_dataset=sentences_dev):
    embedding_matrix, embedding_index = prepare_embedding_matrix(embedding_filename)
    windows_train = flatten(apply_windowing(sentences_train, task, k))
    # determine the tagset from the training data
    list_of_tags = list({tag for word, tag in windows_train})
    model = build_model(k, embedding_matrix, list_of_tags)
    train_x, train_y = encode_data(windows_train, k, list_of_tags, embedding_index)

    history = model.fit(train_x, train_y,
                        batch_size=200,
                        epochs=10)

    # predict on the dev set
    windows_dev = flatten(apply_windowing(sentences_dev, task, k))
    dev_x, _ = encode_data(windows_dev, k, list_of_tags, embedding_index)

    predictions_dev_onehot = model.predict(dev_x, batch_size=1)
    predictions_dev = decode_output(predictions_dev_onehot, list_of_tags)

    true_tags_dev = [tag for words, tag in windows_dev]
   # print('predictions_dev ==', predictions_dev)

    # compute evaluation metrics
    acc = accuracy(true_tags_dev, predictions_dev)
   # print('list_of_tags', list_of_tags)
    f1 = F1(true_tags_dev, predictions_dev, list_of_tags)


    return acc, f1

acc, f1 = run(task="pos", k=3, embedding_filename="hex05.word2vec.300d.txt")

print('acc =', acc)
print('f1 = ', f1)
# windows_train = apply_windowing(sentences_train, 'pos', 3)
# windows_train = flatten(windows_train)
# list_of_tags = list({tag for word, tag in windows_train})
# encode_data(windows_train,3,list_of_tags,word_to_embedding_index)
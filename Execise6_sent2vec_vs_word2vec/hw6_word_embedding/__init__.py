import numpy as np
import tensorflow as tf

dev_review_path = "../hex06_data/rt-polarity.dev.reviews.txt"
dev_label_path = "../hex06_data/rt-polarity.dev.labels.txt"
test_label_path = "../hex06_data/rt-polarity.test.labels.txt"
test_review_path = "../hex06_data/rt-polarity.test.reviews.txt"
train_label_path = "../hex06_data/rt-polarity.train.labels.txt"
train_review_path = "../hex06_data/rt-polarity.train.reviews.txt"


#===============================part 1========================================#

#==================reader================================#
def load_reviews_and_labels(reviews_path, labels_path):
    #==============change label to onehot================#
    def labels_to_onehot(labels):
        labels_onehot = np.zeros((len(labels),2),dtype=np.int32)
        for i,label in enumerate(labels):
            if label == "POS":
                labels_onehot[i,0] = 1
            else:
                labels_onehot[i,1] = 1
        return labels_onehot
    #==============read file=============================#
    reviews = []
    labels = []
    with open(reviews_path) as rfile, open(labels_path) as lfile:
        for rline, lline  in zip(rfile,lfile):
            reviews.append(rline)
            labels.append(lline)

    return reviews, labels_to_onehot(labels)


#===================load_word2vec_embedding=======================#
#Tips: this word2vec is previous trained through word2vec tools

word2vec_embedding_path = "../word2vec/en_wiki_dump_100k_vec2_2.txt"
def load_word2vec_embeddings(embedding_path):
    embedding_dict = {}
    with open(embedding_path) as file:
        for line in file:
            words_list = line.split(" ")[:-1]
            if len(words_list) < 300:
                continue
            embedding_dict[words_list[0]] = words_list[1:]
    embedding_dimension = len(embedding_dict['the'])
    return embedding_dict, embedding_dimension


#load_word2vec_embeddings(embedding_path)

#===================create_word2vec_embedding for each review sentence, using average method=======================#

def create_word2vec_input(embedding_dict, embedding_dimension, reviews):

    # !!!!! if happen to be a word that does not exist in embedding_dict, we use unknow_embedding
    unknown_embedding = np.zeros(embedding_dimension)
    # create an average embedding for each review sentence
    data_x = np.zeros((len(reviews), embedding_dimension))
    for i, review in enumerate(reviews):
        # collect embedding vecs of each word in the review
        words_in_review = np.zeros((len(review), embedding_dimension))
        for w, word in enumerate(review):
            words_in_review[w] = embedding_dict.get(word,unknown_embedding)
        # average them
        data_x[i] = np.mean(words_in_review,axis=0)
    return data_x

#===================create_sent2vec_embedding for each review sentence=====================#

def create_sent2vec_input(sentence_embedding_path):
    data_x = []
    with open(sentence_embedding_path) as file:
        for line in file:
            if not line:
                continue
            data_x.append([float(s) for s in line.strip().split()])
    return np.array(data_x)

# print("Sent2Vec embedding: {}\n".format(create_sent2vec_input("../hex06_data_solution/s2v.train.vecs.rep")[0,:]))
#
# revs, _ = load_reviews_and_labels("hex06_data/rt-polarity.train.reviews.txt", "hex06_data/rt-polarity.train.labels.txt")
# embed_dict, embed_dim = load_word2vec_embeddings("hex06_data_solution/w2v_100K.txt")
# print("word2vec embedding: {}\n".format(create_word2vec_input(embed_dict, embed_dim, revs)[0,:]))



#===============================part 2========================================#

# using keras to set a model

from keras.models import Sequential
from keras.layers import Dense

epochs = 10
batch_size = 500

def run(embeddings = "sent2vec"):
    train_reviews, train_y = load_reviews_and_labels("../hex06_data/rt-polarity.train.reviews.txt", "../hex06_data/rt-polarity.train.labels.txt")
    dev_reviews, dev_y = load_reviews_and_labels("../hex06_data/rt-polarity.dev.reviews.txt", "../hex06_data/rt-polarity.dev.labels.txt")
    test_reviews, test_y = load_reviews_and_labels("../hex06_data/rt-polarity.test.reviews.txt", "../hex06_data/rt-polarity.test.labels.txt")

    if embeddings == "sent2vec":
        train_x = create_sent2vec_input("../hex06_data_solution/s2v.train.vecs.rep")
        dev_x = create_sent2vec_input("../hex06_data_solution/s2v.dev.vecs.rep")
        test_x = create_sent2vec_input("../hex06_data_solution/s2v.test.vecs.rep")
        embedding_dimension = train_x.shape[1]
    elif embeddings == "avg_word2vec":
        embedding_dict, embedding_dimension = load_word2vec_embeddings("../hex06_data_solution/w2v_100K.txt")
        train_x = create_word2vec_input(embedding_dict, embedding_dimension, train_reviews)
        dev_x = create_word2vec_input(embedding_dict, embedding_dimension, dev_reviews)
        test_x = create_word2vec_input(embedding_dict, embedding_dimension, test_reviews)

    number_of_labels = train_y.shape[1]

    print("==========", train_x.shape)
    # create amodel
    model = Sequential()
    model.add(Dense(number_of_labels, activation='softmax', input_shape=(embedding_dimension,)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model = build_model(embedding_dimension, number_of_labels)
    model.fit(train_x,
              train_y,
              batch_size=batch_size,
              epochs=epochs)
    loss, accuracy = model.evaluate(test_x,test_y,batch_size=batch_size)

    return loss, accuracy


for embeddings in ["sent2vec", "avg_word2vec"]:
    loss, accuracy = run(embeddings=embeddings)
    print("\nLoss on test with {} embeddings: {}, accuracy: {}\n".format(embeddings, loss, accuracy))
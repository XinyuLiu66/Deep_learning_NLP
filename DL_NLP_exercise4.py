import numpy as np
import gensim.models.keyedvectors as KeyedVectors
from scipy.stats import spearmanr
#begin=====================read file(combined.tab) from WS-353 =======================================//
path = '/Users/apple/Downloads/wordsim353/combined.tab'
embeddings_path = "/Users/apple/Downloads/GoogleNews-vectors-negative300.bin"
def read(path):
    words_pairs = []
    golden_label = []
    with open(path) as file:
        for line in file:   # read each line
            w1, w2, label = line.split("\t")
            words_pairs.append((w1,w2))
            golden_label.append(label)

    words_pairs = words_pairs[1:]
    golden_label = golden_label[1:]
    for i in range(len(golden_label)):
        golden_label[i] = np.float(golden_label[i])   # using np to change str to float
    return words_pairs,golden_label

#end=====================read file(combined.tab) from WS-353 =======================================//


#begin=====================test similarity =======================================//

words_pairs,golden_label = read(path)
#print("Started reading embeddings...")
word_vectors = KeyedVectors.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
#print("Embeddings read.")
prediction = [word_vectors.similarity(w1,w2) for w1,w2 in words_pairs]
#print(prediction)

#end=====================test similarity =======================================//


#begin=====================spearmann correlation coefficient =======================================//
input_data = input("input five words:")
words_list = input_data.split(" ")

def Word_intrusion(words_list):
    return word_vectors.doesnt_match(words_list)

word_dif = Word_intrusion(words_list)
print(word_dif)



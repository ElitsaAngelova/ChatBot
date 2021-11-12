import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

DATASET_FILE = 'dataset3.json'
DATASET_ENCODING = 'utf8'

MODEL_FILE = 'model2.tflearn'

HIDDEN_LAYER_SIZE = 8
ACTIVATION_FUNCTION = "softmax"
EPOCHS = 1000
BATCH_SIZE = 8

with open(DATASET_FILE, encoding=DATASET_ENCODING) as file:
    data = json.load(file)
	
words_array = []
tags_array = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        current_pattern = nltk.word_tokenize(pattern)
        words_array.extend(current_pattern)
        docs_x.append(current_pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in tags_array:
        tags_array.append(intent["tag"])

words_array = [stemmer.stem(word.lower()) for word in words_array]
words_array = sorted(list(set(words_array)))

tags_array = sorted(tags_array)

training_data = []
output_labels = []

output_labels_empty_row = [0 for _ in range(len(tags_array))]

for x, doc in enumerate(docs_x):
    bag_of_words = []

    current_pattern = [stemmer.stem(word.lower()) for word in doc]

    for word in words_array:
        if word in current_pattern:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)

    output_labels_row = output_labels_empty_row[:]
    output_labels_row[tags_array.index(docs_y[x])] = 1

    training_data.append(bag_of_words)
    output_labels.append(output_labels_row)


training_data = numpy.array(training_data)
output_labels = numpy.array(output_labels)

try:
	tensorflow.reset_default_graph()

	neural_network = tflearn.input_data(shape=[None, len(training_data[0])])
	neural_network = tflearn.fully_connected(neural_network, HIDDEN_LAYER_SIZE)
	neural_network = tflearn.fully_connected(neural_network, HIDDEN_LAYER_SIZE)
	neural_network = tflearn.fully_connected(neural_network, len(output_labels[0]), activation=ACTIVATION_FUNCTION)
	neural_network = tflearn.regression(neural_network)

	model = tflearn.DNN(neural_network)

	model.load(MODEL_FILE)
except:
	tensorflow.reset_default_graph()

	neural_network = tflearn.input_data(shape=[None, len(training_data[0])])
	neural_network = tflearn.fully_connected(neural_network, HIDDEN_LAYER_SIZE)
	neural_network = tflearn.fully_connected(neural_network, HIDDEN_LAYER_SIZE)
	neural_network = tflearn.fully_connected(neural_network, len(output_labels[0]), activation=ACTIVATION_FUNCTION)
	neural_network = tflearn.regression(neural_network)

	model = tflearn.DNN(neural_network)

	model.fit(training_data, output_labels, n_epoch=EPOCHS, batch_size=BATCH_SIZE, show_metric=True)
	model.save("model2.tflearn")
	
def create_bag_of_words(sentence, words_array):
    bag_of_words = [0 for _ in range(len(words_array))]

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]

    for sentence_word in sentence_words:
        for i, word in enumerate(words_array):
            if word == sentence_word:
                bag_of_words[i] = 1
            
    return numpy.array(bag_of_words)

def get_movie(input_sentence):
	results = model.predict([create_bag_of_words(input_sentence, words_array)])
	results_index = numpy.argmax(results)
	tag = tags_array[results_index]

	for intent in data["intents"]:
		if intent['tag'] == tag:
			responses = intent['responses']

	print(random.choice(responses))

"""
def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        input_sentence = input("You: ")
        if input_sentence.lower() == "quit":
            break

        results = model.predict([create_bag_of_words(input_sentence, words_array)])
        results_index = numpy.argmax(results)
        tag = tags_array[results_index]

        for intent in data["intents"]:
            if intent['tag'] == tag:
                responses = intent['responses']

        print(random.choice(responses))
"""
import pandas as pd
import numpy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from weather import get_weather

def get_golf_report():
	golf_file = "dataset2.csv"

	golf_file_handler = open(golf_file, "r")
	golf_data = pd.read_csv(golf_file_handler, sep=",")
	golf_file_handler.close()

	naive_b = BernoulliNB()

	train_features = golf_data.iloc[:,0:7]
	train_label = golf_data.iloc[:,7]

	naive_b.fit(train_features, train_label)

	return naive_b.predict([get_weather()])[0]
    
golf_file = "dataset2.csv"

golf_file_handler = open(golf_file, "r")
golf_data = pd.read_csv(golf_file_handler, sep=",")
golf_file_handler.close()

train, test = train_test_split(golf_data,test_size=0.2, random_state=1)

naive_b = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

train_features = train.iloc[:,0:7]
train_label = train.iloc[:,7]

test_features = test.iloc[:,0:7]
test_label = test.iloc[:,7]

naive_b.fit(train_features, train_label)

test_data = pd.concat([test_features, test_label], axis=1)
test_data["prediction"] = naive_b.predict(test_features)

print("Naive Bayes Accuracy:", naive_b.score(test_features,test_label))

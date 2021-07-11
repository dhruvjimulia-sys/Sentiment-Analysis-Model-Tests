###### Import Libraries ######
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

import spacy
nlp = spacy.load("en_core_web_md")

import pickle

print("Imports complete")

####### Data Preprocessing Functions #######
def is_not_br(token):
    return ((not ('<br' in token.text)) & (not ('/><br' in token.text)) & (not ('/>' in token.text)) & (token.text != 'br') & (not ('<' in token.text)))
def tokenize(text):
    clean_tokens = []
    for token in nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct) & is_not_br(token) & (not token.text.isdigit()):
            clean_tokens.append(token.lemma_.lower())
    return clean_tokens
# Maps 'positive' to 1 and 'negative' to 0
def convert_y(sentiments):
    converted_list = []
    for sentiment in sentiments:
        converted_list.append(1 if sentiment == 'positive' else 0)
    return np.array(converted_list)

###### Loading Data ######
imdb = pd.read_csv('../imdbDataset.csv')
imdb = imdb.iloc[:1500]
pd.set_option("display.max_colwidth", None)
print(f"Length : {len(imdb)}")

print("Data Loading Complete")

##### Word -> Number Encoding ######
##### Bag Of Words
##### Hyperparams: max_features
def bag_of_words(X, y):
    print("Starting Bag of Words Model")
    with open("is_vectorizer_created", "r") as fin:
        if ('False' in fin.read()):
            print("Did not find vectorizer. Creating new vectorizer...")
            bow_transformer = CountVectorizer(analyzer=tokenize, max_features=2000).fit(X)
            pickle.dump(bow_transformer, open("bow_transformer.pickle", "wb"))
        else:
            print("Vectorizer Found. Loading Vectorizer...")
            bow_transformer = pickle.load(open("bow_transformer.pickle", "rb"))
        X = bow_transformer.transform(X)
        y = convert_y(y)
        print("Bag of Words Model Completed")
    with open("is_vectorizer_created", "w") as fout:
        fout.write("True")
    return X, y

##### TF-IDF Vectorization
##### Hyperparams: max_features
def tf_idf(X, y):
    print("Starting TF-IDF Model")
    with open("is_vectorizer_created", "r") as fin:
        if ('False' in fin.read()):
            print("Did not find vectorizer. Creating new vectorizer...")
            tfidf_transformer = TfidfVectorizer(analyzer=tokenize, max_features=2000).fit(X)
            pickle.dump(tfidf_transformer, open("tfidf_transformer.pickle", "wb"))
        else:
            print("Vectorizer Found. Loading Vectorizer...")
            tfidf_transformer = pickle.load(open("tfidf_transformer.pickle", "rb"))            
        X = tfidf_transformer.transform(X)
        y = convert_y(y)
        print("TF-IDF Model Created")
    with open("is_vectorizer_created", "w") as fout:
        fout.write("True")
    return X, y

##### Pre-trained Word Embeddings (Word2Vec Twitter Model)
##### Hyperparams: 
def word2vec(X, y):
    print("Starting Word2Vec Model")
    with open("is_vectorizer_created", "r") as fin:
        if ('False' in fin.read()):
            print("Using pretrained Spacy vectors to train model")
            print("No reviews found. Converting reviews to vectors...")
            vectorized_reviews = []
            for index, review in enumerate(X):
                avg_vector = [0] * 300 
                for token in tokenize(review):
                    avg_vector += nlp(token).vector
                avg_vector = avg_vector / len(X)
                vectorized_reviews.append(avg_vector)
                print(f"Review {index}: Done")
            pickle.dump(vectorized_reviews, open("word2vec_reviews.pickle", "wb"))
        else:
            print("Vectorized reviews found. Loading word vectors...")
            vectorized_reviews = pickle.load(open("word2vec_reviews.pickle", "rb"))
            for index, vector in enumerate(vectorized_reviews):
                vectorized_reviews[index] = vector + 1
    y = convert_y(y)    
    print("Word2Vec Model Created")
    with open("is_vectorizer_created", "w") as fout:
        fout.write("True")
    return vectorized_reviews, y

##### BERT Language Model

##### Model Types ######
##### Logistic Regression
##### Hyperparams: train_test_split
def logistic_regression(X, y):
    print("Starting creation of Logistic Regression Model")
    logistic_model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    print("Logistic Regression Model created")
    return y_test, y_pred

##### K-Nearest-Neighbours Classifier
##### Hyperparams: n_neighbours, train_test_split
def knn_classifier(X, y, number_neighbors):
    print("Started creation of KNN Classifier")
    knn_classifier = KNeighborsClassifier(n_neighbors=number_neighbors)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    print("KNN Classifier created")
    return y_test, y_pred


##### Naive Bayes Classifier
##### Hyperparams: train_test_split
def naive_bayes(X, y):
    print("Started creation of Naive Bayes Classifier")
    nb_model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    print("Naive Bayes Classifier created")
    return y_test, y_pred

##### Applying Models And Printing Accuracy #####
X, y = tf_idf(imdb['review'], imdb['sentiment'])
y_test, y_pred = logistic_regression(X, y)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy: ", round(accuracy, 2), ", Precision: ", round(precision, 2), ", Recall ", round(recall, 2))

##### KNN Classifier Implementation
# max_accuracy = -1
# max_accuracy_neighbors = 0

# for i in range(1, 10):
#     print(f"Number of neighbors: {i}")
#     y_test, y_pred = knn_classifier(X, y, i)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)

#     print("Accuracy: ", round(accuracy, 2), ", Precision: ", round(precision, 2), ", Recall ", round(recall, 2))
#     print("\n")

#     if (accuracy > max_accuracy):
#         max_accuracy = accuracy
#         max_accuracy_neighbors = i

# print(f"Maximum Accuracy obtained was {max_accuracy} with {max_accuracy_neighbors}")
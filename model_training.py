import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
import nltk
import emoji
import regex
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold # import KFold
import pickle

import sys

#Install nltk packages
nltk.download('punkt')
nltk.download('words')
nltk.download('stopwords')

#define sets of english words and stop words
english_words = set(nltk.corpus.words.words())
stop_words = stopwords.words('english')

words_to_remove = {'english':english_words,'stop':stop_words}

#Functions to clean the dataset:
def mapSentiment(label):
    '''
    Takes a label and outputs the english version of the label.
    '''
    sentiments = {0:'negative',4:'positive',2:'neutral'}
    return sentiments[label]

def CleanText(tweet):
    '''
    Takes a normal tweet, lowers it, extracts the english translation of any emoji,
    removes punctuation, removes non-English characters, and tokenizes the tweet.
    '''
    #Lower the tweet
    tweet = tweet.lower()
    #For emojis, replace with the text equivalent.
    tweet = emoji.demojize(tweet)
    #Remove all punctuation
    tweet = re.sub(r'[^\w\s]','',tweet)
    #Tokenize the tweet
    tweet = word_tokenize(tweet)
    return tweet

#Reads input file from command line argument
input_file = sys.argv[1]
print("Reading file...")
tweetset = pd.read_csv(input_file, encoding='latin-1', names = ['sentiment','id','date','flag','user','text'],\
                       usecols = ['sentiment','text'])

print("Input file read...")

#Start cleaning the dataset
#Create Sentiment field with english sentiment
tweetset['Sentiment'] = tweetset.sentiment.apply(mapSentiment)
#Tokenize tweets

#Create pipeline to do all the pre-Preprocessing
class CleanTweets(TransformerMixin):
    def __init__(self,words_to_remove):
        self.words_to_remove = words_to_remove
        #self.stop_words = words_to_remove['stop']
        pass

    def transform(self,X,y=None):
        print("Tokenizing Tweets...")
        X['text'] = X.apply(CleanText)
        #Remove non-english words
        print("Removing non-english words...")
        X['text'] = X.apply(lambda tweet: " ".join(w for w in tweet if w in self.words_to_remove['english']))
        #Remove stop words
        print("Removing Stop Words...")
        X['text'] = X.apply(lambda tweet: " ".join(w for w in tweet.split(' ') if w not in self.words_to_remove['stop']))
        return X

    def fit(self,X,y):
        return self



#Model Training
X = tweetset['text']
y = tweetset['sentiment']
#Split the dataset into training and test
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

#The pipeline turn the tokens into a sparse matrix of words, and gets the tfidf ratio for the words
#Then it feeds this dataset into a classifer
pipeline = Pipeline([
    ('cleaner',CleanTweets(words_to_remove)), #Cleans the dataset
    ('bow',CountVectorizer()),  # strings to token integer counts, analyzer=text_processing
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    #('clf',SGDClassifier(tol=1e-3))
])

print("Training Model...")
pipeline.fit(X_train,y_train)
print("Generating Predictions...")
predictions = pipeline.predict(X_test)
print(classification_report(predictions,y_test))
print(confusion_matrix(predictions,y_test))
print(accuracy_score(predictions,y_test))

#Save the model with the name already given in command linear_model
pickle.dump(pipeline, open(sys.argv[2], 'wb'))
print("Model Saved at "+sys.argv[2])

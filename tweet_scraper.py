'''
A script to download and store all tweets related to a particular topic that the user chooses.
Accesses the Twitter API using a package called tweepy
Stores the tweets in a json file format
Worked on by Aneesh Muthiyan and Ayushi Shrivastav.
Code partially taken from https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
'''


#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import KFold # import KFold

import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import emoji
import regex

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


from tweepy import Stream
from tweepy.streaming import StreamListener

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import pickle
import pandas as pd
import sys
import tweepy
import json
from tweepy import OAuthHandler

def CleanText(tweet):
    #Lower the tweet
    tweet = tweet.lower()

    #For emojis, replace with the text equivalent.
    tweet = emoji.demojize(tweet)

    #Remove all punctuation
    tweet = re.sub(r'[^\w\s]','',tweet)

    #Tokenize the tweet
    tweet = word_tokenize(tweet)

    return tweet


#Setup the authentication with codes provided from Twitter
consumer_key = 'WgRr0afbAzn9YqYyYoET9ltX7'
consumer_secret = 'qEYUZiTafgAWyCal8EV2O0jPRr5hQuI9rRgEVbCto5kYioHgij'
access_token = '1109481200262045698-dg6hGy9rfiKZX779CpuoeqkeboqqOQ'
access_secret = '5Z4Xpi2wm3wyYRcuHyz4hhbGeJkDrSDe8Eu5aEK3nzaob'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

#Setup a Stream Listener to track all tweets related to a topic.
#Topic will be provided by user as command line argument

topic = sys.argv[1]

#Get Historical tweets
date = sys.argv[2]
api = tweepy.API(auth,wait_on_rate_limit=True)

f = open(topic.strip('#')+'.json', 'a')
for tweet in tweepy.Cursor(api.search,q=topic,count=1,lang="en",since=date).items():
    f.write(json.dumps(tweet._json)+'\n')
    print(json.dumps(tweet._json))
f.close()

# read the entire file into a python array
url = topic.strip('#')+'.json'

# now, load it into pandas
df = pd.read_json(url, lines = True)
print(df)
tweetset = df['text']

english_words = set(nltk.corpus.words.words())
stop_words = stopwords.words('english')

tweetset = tweetset.apply(CleanText)
tweetset= tweetset.apply(lambda tweet: " ".join(w for w in tweet if w in english_words))
tweetset = tweetset.apply(lambda tweet: " ".join(w for w in tweet.split(' ') if w not in stop_words))

print(tweetset)
filename = 'finalized_model.pkl'
# load the model from disk
pipeline = pickle.load(open(filename, 'rb'))

predictions = pipeline.predict(tweetset)
print(predictions)
predict_df = pd.DataFrame({'sentiment':predictions,'text':tweetset},columns=['sentiment','text'])
print(predict_df)

#Create countplot of tweets vs sentiments
count_plot = sns.countplot(x='sentiment',data=predict_df)
fig = count_plot.get_figure()
fig.savefig(url+"_countplot.png")

sentiment_dict = {4:'',0:''}
for sentiment in sentiment_dict:
  sentiment_set = predict_df[predict_df['sentiment'] == sentiment]
  sentiment_dict[sentiment] =' '.join(sentiment_set['text'])

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(sentiment_dict[4])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig(url+"_positive.png")

try:
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(sentiment_dict[0])
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(url+"_negative.png")
except ValueError:
    print("No negative sentiments")

'''
#Stream Listener
class MyListener(StreamListener):

    def on_data(self, tweet):
        try:
            with open(topic.strip('#')+'.json', 'a') as f:
                #print(tweet)
                f.write(tweet)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[topic])
'''

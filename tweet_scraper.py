'''
A script to download and store all tweets related to a particular topic that the user chooses.
Accesses the Twitter API using a package called tweepy
Stores the tweets in a json file format
Worked on by Aneesh Muthiyan and Ayushi Shrivastav.
Code partially taken from https://marcobonzanini.com/2015/03/02/mining-twitter-data-with-python-part-1/
'''
import sys
import tweepy
import json
from tweepy import OAuthHandler

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
from tweepy import Stream
from tweepy.streaming import StreamListener

topic = sys.argv[1]

#Get Historical tweets
date = sys.argv[2]
api = tweepy.API(auth,wait_on_rate_limit=True)

f = open(topic.strip('#')+'.json', 'a')
for tweet in tweepy.Cursor(api.search,q=topic,count=100,lang="en",since=date).items():
    f.write(json.dumps(tweet._json)+'\n')
    print(json.dumps(tweet._json))
f.close()



#Stream Listener
class MyListener(StreamListener):

    def on_data(self, tweet):
        try:
            with open(topic.strip('#')+'.json', 'a') as f:
                print(tweet)
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

#%%
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle
import flair
import random
import pprint as pprint
#%%
# Load Data 
# my_tokens = pickle.load(open('./data-extended/my_tokens.pkl', "rb" ))
my_df = pd.read_csv('./data-extended/tweets_df.csv')
"""
To cut short the compute time, I am only going to run 10% of the data
Flair sentiment analysis (~ 7 sentences/s) is orders of magnitude slower than VADER (2400 sentences/s)
I could possibly do this on AWS or Google Compute, but I have run out of free credits on both...
"""
my_df = my_df.sample(frac=0.1, random_state=123)
#%%
my_df.columns
tmp = ['Unnamed: 0', 'id', 'conversation_id', 'created_at', 'date', 'time',
       'timezone', 'user_id', 'username', 'name', 'place', 'tweet', 'language',
       'mentions', 'urls', 'photos', 'replies_count', 'retweets_count',
       'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
       'video', 'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest']
for tm in tmp:
    print(tm)
# %%
# Defining the ETL functions...
def run_vader_analysis(tweets):
    '''
    Use VADER to get sentiment.
    '''
    #print('\nDoing VADER sentiment analysis\n')
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for tweet in tqdm(tweets):
        vader_sentiment = analyzer.polarity_scores(tweet)
        sentiments.append(vader_sentiment)
        # print(f'{tweet[:50]}........{vader_sentiment}')
    return sentiments

def run_flair_analysis(tweets):
    # Instantiate Model
    flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
    sentiments = []
    for tweet in tqdm(tweets):
        s = flair.data.Sentence(tweet)
        flair_sentiment.predict(s)
        total_sentiment = s.labels
        sentiments.append(total_sentiment)
    return sentiments

def main(inputdf, dry_run=False):
    # Run analysis on small subset
    if dry_run:
        inputdf = inputdf.head(n=1000)
    vader_sents = run_vader_analysis(inputdf['tweet'])
    flair_sents = run_flair_analysis(inputdf['tweet'])
    
    return vader_sents, flair_sents

# %%
if __name__ == '__main__':
    vader_results, flair_results = main(my_df, dry_run=False)
# %%
my_df['VADER'] = vader_results
my_df['Flair'] = flair_results
# %%
my_df.to_csv('./data-extended/tweets-results.csv')
# %%

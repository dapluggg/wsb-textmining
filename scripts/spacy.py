#%%
from logging import StringTemplateStyle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import emoji
from tqdm import tqdm
import nltk
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import pprint
import pickle
'''
Download required spacy pipeline. en_core_web_sm is the english pipleline
Available at: https://spacy.io/models/en
'''
nlp = spacy.load("en_core_web_sm")

# %%
reddit = pd.read_csv('reddit_wsb.csv', parse_dates=['timestamp'])
reddit['body'] = reddit['body'].astype(str)
#%%
# Remove the first tweet, it is an outlier 
START_DATE = '2021-01-01'
reddit = reddit.loc[reddit['timestamp'] >= START_DATE]


# %%
reddit.info()
# %%
reddit.head()
# %%
posts_per_day = reddit.timestamp.dt.date.value_counts()
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.lineplot(x=posts_per_day.index, y=posts_per_day)
ax.set(ylabel='# of Posts', title='r/wallstreetbets: Posts Per Day')
#sns.barplot(x='timestamp', data=posts_per_day)
# %%
posts_per_dayname = reddit.timestamp.dt.day_name().value_counts()
sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x=posts_per_dayname.index, y=posts_per_dayname,
order=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
ax.set(ylabel='# of Posts', title='r/wallstreetbets: Posts Per Day of Week')

# %%
# Defining the ETL functions...
def clean_text(text_series, emoji_toText=False):
    '''
    Remove Emojis, replace with text
    Remove mentions
    Remove URLs
    Append itme to cleaned text
    emoji_toText: (default - False) Convert emoji to text using the emoji package. 
    '''
    #print('\nCleaning Text\n')
    cleaned_text = []
    for item in tqdm(text_series):
        
        # To remove capitalization, I could use this.
        # But let's keep this a hidden feature for now
        # item = item.lower()
        if (emoji_toText):
            # Remove Emojis, replace with text
            item = emoji.demojize(item, delimiters=(' ', ''))
        # Remove mentions
        item = re.sub('@[^\s]+', '', item)
        # Remove URLs
        item = re.sub(r"http\S+", '', item)
        if (item=='nan'):
            cleaned_text.append('')
        else: 
            # Append itme to cleaned text
            cleaned_text.append(item)    
    return cleaned_text

def tokenize_posts(text_series):
    '''
    Using TweetTokenizer, get tokens.
    '''
    #print('\nTokenizing text\n')
    tokenizer = TweetTokenizer()
    tokens = []
    for post in tqdm(text_series):
        tokens.append(tokenizer.tokenize(post))
    return tokens

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

def run_spacy_ner(tweets):
    '''
    Run the Spacy Pipleine (en_core_web_sm). English pipeline optimized for CPU. 
    Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
    '''
    #print('\nRunning Spacy Pipline for NER\n')
    named_entities = []
    for tweet in tqdm(tweets):
        doc = nlp(tweet)
        #pprint.pprint(doc)
        named_entities.append(doc)
    return named_entities

def main(inputdf, dry_run=False, save_output=True):
    # Run analysis on a small subset of data 
    # Saves time 
    if dry_run:
        inputdf = inputdf.sample(n=250, axis=0)
    
    clean = clean_text(inputdf['title'])
    mytoks = tokenize_posts(clean)
    sentiment_analysis = run_vader_analysis(clean)
    named_ents = run_spacy_ner(clean)
    # Save All the results as a pickle file
    results = {
        'Clean_Sentences': clean,
        'Tokens': mytoks,
        'Sentiments': sentiment_analysis,
        "Named_Entities": named_ents
    }
    if save_output:
        with open('results.pkl', 'wb') as f:
            pickle.dump(my_nlp_pipeline, f)
        f.close()

    return results

# %%
if __name__ == '__main__':
    my_nlp_pipeline = main(reddit, dry_run=True, save_output=False)
# %%

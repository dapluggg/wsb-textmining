# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:21:43 2021

@author: green
"""

import pandas as pd
import re
import datetime
import emoji
from textblob import TextBlob
import glob as g
import time

# Multiprocessing.
import swifter
import concurrent.futures
import multiprocessing
import tqdm

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#NLTK IMPORTS
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

# %%

# I set my CPU cores to limit the overhead on the system so I can use my computer while this processes.
NUM_PROCESSES = multiprocessing.cpu_count()
CHUNCK_SIZE = 10000

def cleanPostDf(df):
    # Rename the self text column to body
    df = df.rename(columns={'selftext': 'body'}, inplace=False)

    df['body'] = df['body'].astype(str)
    df['title'] = df['title'].astype(str)

    # Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']

    print("body")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(cleanData, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("title")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title'] = list(tqdm.tqdm(pool.map(cleanData, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize body')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('rocket emoji count body')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['rocket_count'] = list(tqdm.tqdm(pool.map(rocketEmojiCount, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('diamond hands emoji count body')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['diamond_hands_count'] = list(tqdm.tqdm(pool.map(diamondHandsEmojiCount, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize title')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("body filtered")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('title filtered')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Convert epoch
    print("epoch")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['created_utc_datetime'] = list(tqdm.tqdm(pool.map(datetime.date.fromtimestamp, df['created_utc'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    df['doc_type'] = 'wsb_post'

    return df

def cleanCommentsDf(df):
    df['body'] = df['body'].astype(str)

    # Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']

    print("body")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(cleanData, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('rocket emoji count body')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['rocket_count'] = list(tqdm.tqdm(pool.map(rocketEmojiCount, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('diamond hands emoji count body')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['diamond_hands_count'] = list(tqdm.tqdm(pool.map(diamondHandsEmojiCount, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("body filtered")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Convert epoch
    print("epoch")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['created_utc_datetime'] = list(tqdm.tqdm(pool.map(datetime.date.fromtimestamp, df['created_utc'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Remove top-level number from ids
    print("clean id")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['parent_id'] = list(tqdm.tqdm(pool.map(cleanIds, df['parent_id'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("clean link id")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['link_id'] = list(tqdm.tqdm(pool.map(cleanIds, df['link_id'], chunksize=CHUNCK_SIZE), total=df.shape[0]))  # With a progressbar

    df['doc_type'] = 'wsb_comment'

    return df

# Given an emoji, count them.
def rocketEmojiCount(x):
    if (x == 'nan'):
        return 0
    p = re.compile(r'\:rocket\:')
    cnt = len(p.findall(x))
    return cnt

# Given an emoji, count them.
def diamondHandsEmojiCount(x):
    if (x == 'nan'):
        return 0
    p = re.compile(r'\:gem_stone\:\:raising_hands\:')
    cnt = len(p.findall(x))
    return cnt

# For the WSB IDs.
def cleanIds(x):
    return re.sub(r't\d+\_', '', x)

# Clean the WSB post data.
def cleanData(x):
    text = x
    text = re.sub('@[^\s]+', ' ', text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # remove single chars
    text = re.sub(r'[\~\`\!\@\#\%\^\&\*\+\=\{\}\[\]\|\\\:\;\"\<\>\?\,\.\/]',' ',text) # remove punctuation
    #text = re.sub('\w*\d+\w*', ' ', text) # remove words containing numbers
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[\s\t\n\r]+', ' ', text, flags=re.I)
    return text

def cleanBodyFields(df):
    df['body'] = df['body'].str.lower()
    df['body'] = df['body'].str.replace('[\$\(\)]', '', regex=True)
    return df['body']

def cleanFilteredFields(df):
    df['body_filtered'] = df['body_filtered'].str.lower()
    df['body_filtered'] = df['body_filtered'].str.replace('[\$\(\)]', '', regex=True)
    return df['body_filtered']

def cleanTickersFields(df):
    df['body_tickers'] = df['body_tickers'].str.strip()
    df['body_tickers'] = df['body_tickers'].str.replace('[\$\(\)]', '', regex=True)
    return df['body_tickers']

def cleanTitleFields(df):
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].str.replace('[\$\(\)]', '', regex=True)
    return df['title']

def cleanTitleFilteredFields(df):
    df['title_filtered'] = df['title_filtered'].str.lower()
    df['title_filtered'] = df['title_filtered'].str.replace('[\$\(\)]', '', regex=True)
    return df['title_filtered']

def cleanTitleTickersFields(df):
    df['title_tickers'] = df['title_tickers'].str.strip()
    df['title_tickers'] = df['title_tickers'].str.replace('[\$\(\)]', '', regex=True)
    return df['title_tickers']

# Scrub the text of english stopwords from NLTK.
def stopWordFilter(txt):
    t = txt
    for s in ENGLISH_STOP_WORDS:
        pattern = r"\s+" + s + "\s+"
        t = re.sub(pattern, ' ', t)

    return(t)

# Give reddit body, will look for stock tickers.
# Returns empty array if nothing found.
def getTickersByRe(body):
    note_search = re.search(r"\s+(\$([A-Z]{1,4})|\([A-Z]{1,4}\))\s+", body)

    # If the title exists, extract and return it.
    if note_search:
        return note_search.group()
    return ""

# Extract the stock tickers based on company name.
def getTickersByName(text):
    companyNames = {'gamestop': 'GME',
                    'black berry': 'BB',
                    'blackberry': 'BB',
                    'nokia': 'NOK',
                    'silver': 'SLV',
                    'kelloggs': 'K',
                    'tesla':'TSLA',
                    'space':'SPCE'}

    # Loop through looking for the name.
    for k in companyNames:
        pattern = r"\s+"+ k +"\s+"

        # If the title exists, extract and return it.
        if re.search(pattern, text):
            return companyNames[k]

    return ""

# get the stock data to be merged with the WSB content.
def getStockData(on):
    stockLoc = on['folderLoc'] + '\\rawdata\\gme_amc_cleandata.csv'
    stockDf = pd.read_csv(stockLoc)
    gmeDf = stockDf[["Date", "GME RSI", "GME.Open", "GME.High", "GME.Low", "GME.Close", "GME.Volume", "GME.Adjusted"]]
    gmeDf.columns = ["date", "rsi", "open", "high", "low", "close", "volume", "adjusted"]
    gmeDf['ticker'] = 'GME'
    gmeDf = gmeDf.sort_values(by=['date'])
    return gmeDf

# Vadar Sentiment
def vadar_sentiment(text):
    vader_sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
    if vader_sentiment == 0:
        return "neutral"
    elif vader_sentiment['neg'] != 0:
        return "negative"
    elif vader_sentiment['pos'] != 0:
        return "positive"
    return "neutral"

# TB Polarity value.
def tb_polarity(text):
    return TextBlob(text).sentiment.polarity

# TextBlob subjectivity value.
def tb_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# The pipe. ###################################
def runIngest(on):
    resultsFileLoc = on['folderLoc'] + '\\processed\\' + on['job']
    fileLoc = on['filename']# + '\\rawdata\\' + on['filename']
    header = on['header']
    df = pd.read_csv(fileLoc, usecols=header)#, nrows=10000)

    # Clean the text data.
    if (on['job'] == 'wsb_post_results'):
        df = cleanPostDf(df)
    else:
        df = cleanCommentsDf(df)

    # Polarity subjectivity
    print("polarity")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_polarity'] = list(tqdm.tqdm(pool.map(tb_polarity, df['body_filtered'], chunksize=CHUNCK_SIZE), total=df.shape[0]))  # With a progressbar

    print("subjectivity")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_subjectivity'] = list(tqdm.tqdm(pool.map(tb_subjectivity, df['body_filtered'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('vader sentiment')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_vadar_sentiment'] = list(tqdm.tqdm(pool.map(vadar_sentiment, df['body_filtered'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Find stock tickers
    print("extract stock tickers")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_tickers'] = list(tqdm.tqdm(pool.map(getTickersByRe, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("body touch ups.")
    df.body = df.body.str.lower()
    df.body = df.body.str.replace('[\$\(\)]', '', regex=True)
    df.body_filtered = df.body_filtered.str.lower()
    df.body_filtered = df.body_filtered.str.replace('[\$\(\)]', '', regex=True)
    df.body_filtered = df.body_filtered.str.replace('\:\:', ': :', regex=True)
    df.body_tickers = df.body_tickers.str.strip()
    df.body_tickers = df.body_tickers.str.replace('[\$\(\)]', '', regex=True)

    print("merging stock data.")
    # Merge stock data for both GME
    gmeDf = getStockData(on)
    df['created_utc_datetime'] = df['created_utc_datetime'].astype(str)
    gmeDf['date'] = gmeDf['date'].astype(str)
    df = pd.merge(df, gmeDf, left_on=['created_utc_datetime', 'body_tickers'], right_on=['date', 'ticker'], how='left')

    if (on['job'] == 'wsb_post_results'):
        print('title vadar')
        with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
            df['title_vadar_sentiment'] = list(tqdm.tqdm(pool.map(vadar_sentiment, df['title_filtered'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

        print('title tickers ')
        with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
            df['title_tickers'] = list(tqdm.tqdm(pool.map(getTickersByRe, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

        print("title touch ups.")
        df.title = df.title.str.lower()
        df.title = df.title.str.replace('[\$\(\)]', '', regex=True)
        df.title_filtered = df.title_filtered.str.lower()
        df.title_filtered = df.title_filtered.str.replace('[\$\(\)]', '', regex=True)
        df.title_filtered = df.title_filtered.str.replace('\:\:', ': :', regex=True)
        df.title_tickers = df.title_tickers.str.strip()
        df.title_tickers = df.title_tickers.str.replace('[\$\(\)]', '', regex=True)

        titleDf = df[df['title_tickers'] == 'GME']
        titleDf = titleDf.drop(columns=["date", "rsi", "open", "high", "low", "close", "volume", "adjusted", "ticker"])
        titleDf = pd.merge(titleDf, gmeDf, left_on=['created_utc_datetime', 'title_tickers'], right_on=['date', 'ticker'], how='left')

        df = df[df['title_tickers'] != 'GME'] # drop the rows with GME in the title.
        df = pd.concat([df, titleDf])  # Put the removed back with stock data.

    df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.datetime.fromtimestamp(x))
    df[["rsi", "open", "high", "low", "close", "volume", "adjusted"]] = df[["rsi", "open", "high", "low", "close", "volume", "adjusted"]].fillna(value=0)
    df['gain'] = df.close - df.open

    print ("putting to disk.")
    putToDisk(resultsFileLoc, df)

    # return back to main thread
    return {'job': on['job'], 'df': df}

# Puts the posts and comments to both csv and parque.
def putToDisk(resultsFileLoc, df):
    currentTime = time.strftime('%Y%m%d%H%M%S', time.localtime())
    size = df.shape[0]
    step = 100000
    end = step
    count = 1
    for x in range(0, size, step):
        print(x, end)
        d = df[x:end]
        d.to_csv(resultsFileLoc + '_' + currentTime + '_' + str(count) + '.csv', index=False)
        d.to_parquet(resultsFileLoc + '_' + currentTime + '_' + str(count) + '.gzip', compression='gzip')
        end += step
        count += 1

# %%
if __name__ == '__main__':

    folderLoc = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining'
    wsbPosts = 'wallstreetbets_posts*.csv'
    wsbComments = 'wallstreetbets_comments*.csv'

    wsbPostHeaders = ['id', 'author', 'author_premium', 'created_utc', 'domain', 'title', 'selftext', 'subreddit',
                      'subreddit_id', 'num_comments', 'upvote_ratio']
    wsbCommHeaders = ['id', 'parent_id', 'link_id', 'author', 'created_utc', 'body', 'subreddit', 'subreddit_id']

    files = g.glob(folderLoc + '\\rawdata\\' + wsbPosts, recursive=True)
    print(files)
    for f in files:
        # Job definitions
        postDic = {'job': 'wsb_post_results',
                   'folderLoc': folderLoc,
                   'filename': f,
                   'header': wsbPostHeaders}
        print('starting posts')
        print(postDic)
        runIngest(postDic)
        print('completed posts')

    files = g.glob(folderLoc + '\\rawdata\\' + wsbComments, recursive=True)
    print(files)
    for f in files:
        # Job definitions
        commentsDic = {'job': 'wsb_comments_results',
                       'folderLoc': folderLoc,
                       'filename': f,
                       'header': wsbCommHeaders}


        print('starting comments')
        runIngest(commentsDic)
        print('completed comments')
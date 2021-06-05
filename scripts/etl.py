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
from time import gmtime, strftime

# Multiprocessing.
import swifter
import concurrent.futures
import multiprocessing
import tqdm

#NLTK IMPORTS
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')

# %%

# I set my CPU cores to limit the overhead on the system so I can use my computer while this processes.
NUM_PROCESSES = 12 #multiprocessing.cpu_count()
CHUNCK_SIZE = 50000

def cleanPostDf(df):
    # Rename the self text column to body
    df = df.rename(columns={'selftext': 'body'}, inplace=False)

    df['body'] = df['body'].astype(str)
    df['title'] = df['title'].astype(str)

    # Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']

    print("body")
    #df.body = df.body.swifter.apply(lambda x: cleanData(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(cleanData, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("title")
    #df.title = df.title.swifter.apply(lambda x: cleanData(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title'] = list(tqdm.tqdm(pool.map(cleanData, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize body')
    #df.body = df.body.swifter.apply(lambda x: emoji.demojize(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize title')
    #df.title = df.title.swifter.apply(lambda x: emoji.demojize(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("body filtered")
    #df['body_filtered'] = df.body.swifter.apply(lambda x: stopWordFilter(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('title filtered')
    #df['title_filtered'] = df.title.swifter.apply(lambda x: stopWordFilter(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['title_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Convert epoch
    print("epoch")
    #df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.date.fromtimestamp(x))
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
    #df.body = df.body.swifter.apply(lambda x: cleanData(x))
    #df.body = df.body.swifter.apply(lambda x: emoji.demojize(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(cleanData, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('demojize')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body'] = list(tqdm.tqdm(pool.map(emoji.demojize, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("body_filtered")
    #df['body_filtered'] = df.body.swifter.apply(lambda x: stopWordFilter(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_filtered'] = list(tqdm.tqdm(pool.map(stopWordFilter, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Convert epoch
    print("epoch")
    #df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.date.fromtimestamp(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['created_utc_datetime'] = list(tqdm.tqdm(pool.map(datetime.date.fromtimestamp, df['created_utc'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Remove top-level number from ids
    print("clean id")
    #df.parent_id = df.parent_id.swifter.apply(lambda x: re.sub(r't\d+\_', '', x))
    #df.link_id = df.link_id.swifter.apply(lambda x: re.sub(r't\d+\_', '', x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['parent_id'] = list(tqdm.tqdm(pool.map(cleanIds, df['parent_id'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("clean link id")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['link_id'] = list(tqdm.tqdm(pool.map(cleanIds, df['link_id'], chunksize=CHUNCK_SIZE), total=df.shape[0]))  # With a progressbar

    df['doc_type'] = 'wsb_comment'

    return df

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
    text = re.sub('\w*\d+\w*', ' ', text) # remove words containing numbers
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
    stopWords = set(stopwords.words('english'))
    t = txt
    for s in stopWords:
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
    fileLoc = on['folderLoc'] + '\\rawdata\\' + on['filename']
    header = on['header']
    df = pd.read_csv(fileLoc, usecols=header)#, nrows=10000)

    # Clean the text data.
    if (on['job'] == 'wsb_post_results'):
        df = cleanPostDf(df)
    else:
        df = cleanCommentsDf(df)

    # Polarity subjectivity
    print("Polarity")
    #df['body_polarity'] = df.body.swifter.apply(lambda x: TextBlob(x).sentiment.polarity)
    #df['body_subjectivity'] = df.body.swifter.apply(lambda x: TextBlob(x).sentiment.subjectivity)
    #df['body_subjectivity'] = df.body.swifter.apply(lambda x: vadar_sentiment(x))

    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_polarity'] = list(tqdm.tqdm(pool.map(tb_polarity, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))  # With a progressbar

    print("subjectivity")
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_subjectivity'] = list(tqdm.tqdm(pool.map(tb_subjectivity, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print('vader sentiment')
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_vadar_sentiment'] = list(tqdm.tqdm(pool.map(vadar_sentiment, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    # Find stock tickers
    print("extract stock tickers")
    #df['body_tickers'] = df.body.swifter.apply(lambda x: getTickersByRe(x))
    with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
        df['body_tickers'] = list(tqdm.tqdm(pool.map(getTickersByRe, df['body'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

    print("data touch ups.")
    df.body = df.body.str.lower()
    df.body = df.body.str.replace('[\$\(\)]', '', regex=True)
    df.body_filtered = df.body_filtered.str.lower()
    df.body_filtered = df.body_filtered.str.replace('[\$\(\)]', '', regex=True)
    df.body_tickers = df.body_tickers.str.strip()
    df.body_tickers = df.body_tickers.str.replace('[\$\(\)]', '', regex=True)

    print("merging stock data.")
    # Merge stock data for both GME
    gmeDf = getStockData(on)
    df['created_utc_datetime'] = df['created_utc_datetime'].astype(str)
    gmeDf['date'] = gmeDf['date'].astype(str)
    df = pd.merge(df, gmeDf, left_on=['created_utc_datetime', 'body_tickers'], right_on=['date', 'ticker'], how='left')

    if (on['job'] == 'wsb_post_results'):
        #df['title_vadar_sentiment'] = df.title.swifter.apply(lambda x: vadar_sentiment(x))
        print('title vader')
        with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
            df['title_vadar_sentiment'] = list(tqdm.tqdm(pool.map(vadar_sentiment, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

        # df['title_tickers'] = df.title.swifter.apply(lambda x: getTickersByRe(x))
        print('title tickers ')
        with concurrent.futures.ProcessPoolExecutor(NUM_PROCESSES) as pool:
            df['title_tickers'] = list(tqdm.tqdm(pool.map(getTickersByRe, df['title'], chunksize=CHUNCK_SIZE), total=df.shape[0]))

        df['title_tickers'] = df['title_tickers'].str.strip()
        df.title = df.title.str.lower()
        df.title = df.title.str.replace('[\$\(\)]', '', regex=True)
        df.title_filtered = df.title_filtered.str.lower()
        df.title_filtered = df.title_filtered.str.replace('[\$\(\)]', '', regex=True)
        df.title_tickers = df.title_tickers.str.replace('[\$\(\)]', '', regex=True)

        titleDf = df[df['title_tickers'] == 'GME']
        titleDf = titleDf.drop(columns=["date", "rsi", "open", "high", "low", "close", "volume", "adjusted", "ticker"])
        titleDf = pd.merge(titleDf, gmeDf, left_on=['created_utc_datetime', 'title_tickers'], right_on=['date', 'ticker'], how='left')

        df = df[df['title_tickers'] != 'GME'] # drop the rows with GME in the title.
        df = pd.concat([df, titleDf])  # Put the removed back with stock data.

    df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.datetime.fromtimestamp(x))
    df[["rsi", "open", "high", "low", "close", "volume", "adjusted"]] = df[["rsi", "open", "high", "low", "close", "volume", "adjusted"]].fillna(value=0) 
    #df = df.sort_values(by=['created_utc_datetime'])

    print ("putting to disk.")
    putToDisk(resultsFileLoc, df)

    # return back to main thread
    return {'job': on['job'], 'df': df}

# Puts the posts and comments to both csv and parque.
def putToDisk(resultsFileLoc, df):
    size = df.shape[0]
    step = 250000
    end = step
    count = 1
    for x in range(0, size, step):
        print(x, end)
        d = df[x:end]
        d.to_csv(resultsFileLoc + '_' + str(count) + '.csv', index=False)
        d.to_parquet(resultsFileLoc + '_' + str(count) + '.gzip', compression='gzip')
        end += step
        count += 1

# %%
if __name__ == '__main__':

    folderLoc = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining'
    wsbPosts = 'wallstreetbets_posts.csv'
    wsbComments = 'wallstreetbets_comments.csv'

    wsbPostHeaders = ['id', 'author', 'author_premium', 'created_utc', 'domain', 'title', 'selftext', 'subreddit',
                      'subreddit_id', 'num_comments', 'upvote_ratio']
    wsbCommHeaders = ['id', 'parent_id', 'link_id', 'author', 'created_utc', 'body', 'subreddit', 'subreddit_id']

    # Job definitions
    postDic = {'job': 'wsb_post_results',
               'folderLoc': folderLoc,
               'filename': wsbPosts,
               'header': wsbPostHeaders}

    commentsDic = {'job': 'wsb_comments_results',
                   'folderLoc': folderLoc,
                   'filename': wsbComments,
                   'header': wsbCommHeaders}

    print('starting posts')
    runIngest(postDic)
    print('completed posts')
    print('starting comments')
    runIngest(commentsDic)
    print('completed comments')

    # The jobs to send to the thread pool.
    #jobs = [commentsDic]
    #job_list = []
    #from multiprocessing import Pool

    # Multi process the data.
    #with Pool(len(jobs)) as p:
    #    print('start', strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
    #    job_list = p.map(runIngest, jobs)
    #    #print(job_list)
    #    print('done', strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
    # %%
    #wsbPostDf = pd.DataFrame()
    #wsbCommentsDf = pd.DataFrame()
    #for job in job_list:
    #    if (job['job'] == 'wsb_post_results'):
            #print('wsb_post_results')
    #        wsbPostDf = job['df']
    #    else:
            #print('wsb_comments_results')
    #        wsbCommentsDf = job['df']

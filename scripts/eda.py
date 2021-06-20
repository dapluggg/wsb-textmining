import pandas as pd
import pandasql as pdsql
import seaborn as sns
import glob as g
import matplotlib.pyplot as plt
import datetime
import numpy as np
import re
import concurrent.futures
import tqdm

#NUM_PROCESSES = 12
#CHUNCK_SIZE = 10000

# Read the data from parquet
def getData ():
    filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210619\\wsb_*.csv'
    files = g.glob(filePath, recursive=True)
    postDf = pd.DataFrame()
    for f in files:
        print('processing file:', f)
        df = pd.read_csv(f)
        df.body = df.body.astype(str)
        print(df.shape)
        postDf = postDf.append(df, ignore_index=True)
    print(postDf.shape)
    return postDf

# Print the length statistics.  
def titleLenStats(df):
    df['title_length'] = df['title'].str.len()
    print('===== WSB Title Stats =====')
    print(df['title_length'].describe())
    print('Median title length: ', df['title_length'].median())
    
def bodyLenStats(df):
    df['body_length'] = df['body'].str.len()
    print('===== WSB Body Stats =====')
    print(df['body_length'].describe())
    print('Median body length: ', df['body_length'].median())

def main():
    # Get data into DFs.
    postDf = getData()
    print(postDf)
    
    titleLenStats(postDf)
    bodyLenStats(postDf)

    print("Emoji Stats")
    print(postDf[['body_polarity','body_subjectivity','rocket_count','diamond_hands_count','gain']].describe())
    
    # How do sentiment correlate to stock data.  
    corrDf = postDf[['body_polarity','body_subjectivity','rocket_count','diamond_hands_count','rsi','close','gain','volume','upvote_ratio','num_comments']]
    corrDf = corrDf[(corrDf.rsi > 0 ) & (corrDf.volume > 0 ) & (corrDf.close > 0 )]
    corr = corrDf.corr()
    fig = plt.figure()
    sns.set(style=None)
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(corrDf.columns),1)
    ax.set_xticks(ticks)
    plt.xticks(rotation=90)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corrDf.columns)
    ax.set_yticklabels(corrDf.columns)
    #ax.set_xlabel('Correlation Between Sentiment and Stock Info')
    plt.show()
    print(corr.shape)
# %%
    # Sentiment distribution
    dd = pdsql.sqldf(
        'select body_vadar_sentiment, sum(rocket_count) rocket_freq from postDf group by body_vadar_sentiment order by body_vadar_sentiment asc')
    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x="body_vadar_sentiment", y="rocket_freq", data=dd)
    plt.show()
    
    '''
    # How is volume related to post volume.  
    sdf = postDf[['created_utc','volume','author']]
    sdf['utc_date'] = sdf.created_utc.apply(lambda x: datetime.date.fromtimestamp(x))
    sdf = sdf[sdf['volume'] != 0]
    q1 = 'SELECT volume, count(author) as post_count from sdf GROUP BY utc_date'
    dd = pdsql.sqldf(q1)
    print(dd.shape)
    sns.scatterplot(data=dd, x="volume", y="post_count")
    plt.show()

    # Sentiment distribution
    dd = pdsql.sqldf('select body_vadar_sentiment, doc_type, count(body_vadar_sentiment) sentiment_count from postDf group by body_vadar_sentiment, doc_type order by body_vadar_sentiment asc')
    plt.figure(figsize=(8, 8))
    ax = sns.barplot(x="body_vadar_sentiment", y="sentiment_count", hue='doc_type', data=dd)
    plt.show()
    '''

if __name__ == '__main__':
    print("Executing batch process.")
    main()
    print("Batch process done.")
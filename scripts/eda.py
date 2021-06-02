import pandas as pd
import pandasql as pdsql
import seaborn as sns
import glob as g
import matplotlib.pyplot as plt
import datetime
import numpy as np

# Read the data from parquet
def getData (filePath, nrows):
    glDf = pd.read_csv(filePath, nrows=nrows)
    return glDf

def main():
    # File path.
    postPath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\wsb_post_results.csv'
    commPath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\wsb_comments_results.csv'

    # Get data into DFs.
    postDf = getData(postPath, 9000000)
    commDf = getData(commPath, 10000000)

    # Blend the data frames.
    df = pd.concat([postDf, commDf], sort=False)

    # Correlate subjectivity and polarity with rsi, close, volumne
    sdf = df[['body_polarity','body_subjectivity','close','doc_type']]
    sns.scatterplot(data=sdf, x="body_polarity", y="body_subjectivity", 
                    size='close', sizes=(20,200))
    plt.show()
    
    # How do sentiment correlate to stock data.  
    corrDf = df[['body_polarity', 'body_subjectivity','rsi','close','volume']]
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
    ax.set_xlabel('Correlation Between Sentiment and Stock Info')
    plt.show()
    print(corr.shape)
    
    
    # How is volume related to post volume.  
    sdf = df[['created_utc','volume','author']]
    sdf['utc_date'] = sdf.created_utc.apply(lambda x: datetime.date.fromtimestamp(x))
    sdf = sdf[sdf['volume'] != 0]
    q1 = 'SELECT volume, count(author) as post_count from sdf GROUP BY utc_date'
    dd = pdsql.sqldf(q1)
    print(dd.shape)
    sns.scatterplot(data=dd, x="volume", y="post_count")
    plt.show()

if __name__ == '__main__':
    print("Executing batch process.")
    main()
    print("Batch process done.")
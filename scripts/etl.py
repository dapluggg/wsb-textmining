# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:21:43 2021

@author: green
"""

import pandas as pd
import re
import datetime

# %%
def cleanPostDf(df):
    # Rename the self text column to body
    df = df.rename(columns={'selftext': 'body'}, inplace=False)

    df['body'] = df['body'].astype(str)
    df['title'] = df['title'].astype(str)

    # Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']

    # Remove handlers
    df.title = df.title.apply(lambda x: re.sub('@[^\s]+', '', x))
    df.body = df.body.apply(lambda x: re.sub('@[^\s]+', '', x))

    # Remove URLS
    df.title = df.title.apply(lambda x: re.sub(r"http\S+", "", x))
    df.body = df.body.apply(lambda x: re.sub(r"http\S+", "", x))

    # Remove all the special characters
    df.title = df.title.apply(lambda x: re.sub(r'[^a-zA-Z\$\(\)\s\t]', '', x))
    df.body = df.body.apply(lambda x: re.sub(r'[^a-zA-Z\$\(\)\s\t]', '', x))

    # remove all single characters
    df.title = df.title.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))
    df.body = df.body.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    df.title = df.title.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
    df.body = df.body.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

    # Convert epoch
    df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.datetime.fromtimestamp(x))
    # df[['created_utc_date','created_utc_time']] = df.created_utc.str.split(expand=True)

    df['doc_type'] = 'wsb_post'

    return df


def cleanCommentsDf(df):
    df['body'] = df['body'].astype(str)

    # Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']

    # Remove handlers
    df.body = df.body.apply(lambda x: re.sub('@[^\s]+', ' ', x))

    # Remove URLS
    df.body = df.body.apply(lambda x: re.sub(r"http\S+", " ", x))

    # Remove all the special characters
    df.body = df.body.apply(lambda x: re.sub(r'[^a-zA-Z\$\(\)\s\-\_]', '', x))

    # remove all single characters
    df.body = df.body.apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    df.body = df.body.apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.I))

    # Convert epoch
    df['created_utc_datetime'] = df.created_utc.apply(lambda x: datetime.datetime.fromtimestamp(x))
    # df[['created_utc_date','created_utc_time']] = df.created_utc.str.split(expand=True)

    # Remove top-level number from ids
    df.parent_id = df.parent_id.apply(lambda x: re.sub(r't\d+\_', '', x))
    df.link_id = df.link_id.apply(lambda x: re.sub(r't\d+\_', '', x))

    df['doc_type'] = 'wsb_comment'

    return df


# Give reddit body, will look for stock tickers.
# Returns empty array if nothing found.
def getTickersByRe(body):
    note_search = re.search(r"\s+(\$([A-Z]{1,4})|\([A-Z]{1,4}\))\s+", body)

    # If the title exists, extract and return it.
    if note_search:
        return note_search.group()
    return ""

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
        if (re.search(pattern, text) == True):
            return companyNames[k]

    return ""

def runIngest(on):
    resultsFileLoc = on['folderLoc'] + '\\processed\\' + on['job']
    fileLoc = on['folderLoc'] + '\\rawdata\\' + on['filename']
    header = on['header']
    df = pd.read_csv(fileLoc, usecols=header)#, nrows=100)

    # Clean the text data.
    if (on['job'] == 'wsb_post_results'):
        df = cleanPostDf(df)
    else:
        df = cleanCommentsDf(df)

    # Find stock tickers
    df['body_tickers'] = df.body.apply(lambda x: getTickersByRe(x))
    #df['body_tickers'] = df[df['body_tickers'].str.len() <= 0]['body'].apply(lambda x: getTickersByName(x))

    df.body = df.body.str.lower()
    df.body = df.body.str.replace('[\$\(\)]', '', regex=True)
    df.body_tickers = df.body_tickers.str.replace('[\$\(\)]', '', regex=True)

    if (on['job'] == 'wsb_post_results'):
        df['title_tickers'] = df.title.apply(lambda x: getTickersByRe(x))
        #df['title_tickers'] = df[df['body_tickers'].str.len() <= 0]['title'].apply(lambda x: getTickersByName(x))
        df.title = df.title.str.lower()
        df.title = df.title.str.replace('[\$\(\)]', '', regex=True)
        df.title_tickers = df.title_tickers.str.replace('[\$\(\)]', '', regex=True)

    # Write results to file.
    df.to_csv(resultsFileLoc + '.csv', index=False)
    df.to_parquet(resultsFileLoc + '.gzip', compression='gzip')

    # return back to main thread
    return {'job': on['job'], 'df': df}

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

    # The jobs to send to the thread pool.
    jobs = [postDic, commentsDic]
    job_list = []
    from multiprocessing import Pool

    # Multi process the data.
    with Pool(len(jobs)) as p:
        print('start')
        job_list = p.map(runIngest, jobs)
        #print(job_list)
        print('done')
    # %%
    wsbPostDf = pd.DataFrame()
    wsbCommentsDf = pd.DataFrame()
    for job in job_list:
        if (job['job'] == 'wsb_post_results'):
            #print('wsb_post_results')
            wsbPostDf = job['df']
        else:
            #print('wsb_comments_results')
            wsbCommentsDf = job['df']

    #print(wsbPostDf[['body_tickers', 'body']])
    #print(wsbPostDf[['title_tickers', 'title']])
    #print(wsbCommentsDf[['body_tickers', 'body']])
    # %%
    # Merge the comments with the original post.  Let the comments drive the
    # number of records.

    # %%
    #d = wsbPostDf[wsbPostDf['body_tickers'].apply(len) > 0]
    #dd = pdsql.sqldf('select body_tickers as stock, count(body_tickers) as cnt from d group by body_tickers order by cnt desc')
    # print(dd)

    #ax = sns.barplot(x="stock", y="cnt", data=dd[:20])
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    #ax.set_xlabel('Equity Symbol')
    #ax.set_ylabel('Frequency')
    #ax.set_title('Top 20 Most Found Equity')
    #plt.show()
    # %%
    #dp = wsbPostDf[['author', 'id', 'title']]
    #dc = wsbCommentsDf[['author', 'id', 'parent_id', 'link_id']]
    #print(dp.head(5))
    #print(dc.head(50))
    #ddp = pdsql.sqldf('select author, count(author) as auth_cnt from d group by author order by auth_cnt desc')
    #print(ddp)
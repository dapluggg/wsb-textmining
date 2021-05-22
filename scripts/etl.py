# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:21:43 2021

@author: green
"""

import pandas as pd
import re

# %%
def cleanPostDf(df):
    #Rename the self text column to body
    df = df.rename(columns = {'selftext':'body'}, inplace = False)
    
    df['body'] = df['body'].astype(str)
    df['title'] = df['title'].astype(str)
        
    #Convert to lower case.
    #df.title = df.title.str.lower()
    #df.body = df.body.str.lower()
    
    #Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']
    #df = df[df['body'] != '[removed]']

    #Remove handlers
    df.title = df.title.apply(lambda x:re.sub('@[^\s]+','',x))
    df.body = df.body.apply(lambda x:re.sub('@[^\s]+','',x))

    # Remove URLS
    df.title = df.title.apply(lambda x:re.sub(r"http\S+", "", x))
    df.body = df.body.apply(lambda x:re.sub(r"http\S+", "", x))

    # Remove all the special characters
    df.title = df.title.apply(lambda x:re.sub(r'[^a-zA-Z\$\(\)\s\t]', '', x))
    df.body = df.body.apply(lambda x:re.sub(r'[^a-zA-Z\$\(\)\s\t]', '', x))

    #remove all single characters
    df.title = df.title.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    df.body = df.body.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    df.title = df.title.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    df.body = df.body.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    
    return df

def cleanCommentsDf(df):    
    df['body'] = df['body'].astype(str)
           
    #Convert to lower case.
    #df.title = df.title.str.lower()
    #df.body = df.body.str.lower()
    
    #Drop nulls
    df = df.dropna(subset=['body'])
    df = df[df['body'] != 'nan']
    #df = df[df['body'] != '[removed]']

    #Remove handlers
    df.body = df.body.apply(lambda x:re.sub('@[^\s]+','',x))

    # Remove URLS
    df.body = df.body.apply(lambda x:re.sub(r"http\S+", "", x))

    # Remove all the special characters
    df.body = df.body.apply(lambda x:re.sub(r'[^a-zA-Z\$\(\)\s\t]', '', x))

    #remove all single characters
    df.body = df.body.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))

    # Substituting multiple spaces with single space
    df.body = df.body.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    
    return df

# Give reddit body, will look for stock tickers.
# Returns empty array if nothing found. 
def getTickers(body):
    note_search =  re.search(r"\s+(\$([A-Z]{1,4})|\([A-Z]{1,4}\))\s+", body)
     
    # If the title exists, extract and return it.
    if note_search:
        return note_search.group()
    return ""

def runIngest(on):
    resultsFileLoc = on['folderLoc'] + '\\' + on['job']
    fileLoc = on['folderLoc'] + '\\' + on['filename']
    header = on['header']
    df = pd.read_csv(fileLoc, usecols = header)
        
    # Clean the text data.
    if (on['job'] == 'wsb_post_results'):
        df = cleanPostDf(df)
    else:
        df = cleanCommentsDf(df)
    
    # Find stock tickers
    df['body_tickers'] = df.body.apply(lambda x : getTickers(x))
    df.body = df.body.str.lower()
    
    if (on['job'] == 'wsb_post_results'): 
        df['title_tickers'] = df.title.apply(lambda x : getTickers(x))
        df.title = df.title.str.lower()
        
    # Write results to file.
    df.to_csv(resultsFileLoc + '.csv')
        
    # return back to main thread
    return df
    
    

# %%
if __name__ == '__main__':
    
    folderLoc = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\data'
    wsbPosts = 'wallstreetbets_posts.csv'
    wsbComments = 'wallstreetbets_comments.csv'
    
    wsbPostHeaders = ['id','author','author_premium','created_utc','domain','title','selftext','subreddit','subreddit_id','num_comments','upvote_ratio']
    wsbCommHeaders = ['id','parent_id','author','created_utc','body','subreddit','subreddit_id']
    
    postDic = {'job':'wsb_post_results',
             'folderLoc':folderLoc,
             'filename':wsbPosts,
             'header':wsbPostHeaders}
    
    commentsDic = {'job':'wsb_comments_results',
                   'folderLoc':folderLoc,
                   'filename':wsbComments,
                   'header':wsbCommHeaders}
    
    jobs = [postDic, commentsDic]
    from multiprocessing import Pool 
    with Pool(len(jobs)) as p:
        print('start')
        df = p.map(runIngest, jobs)
        print(df)
        print('done')
        
    
# %%

 
    
    
  
# %%
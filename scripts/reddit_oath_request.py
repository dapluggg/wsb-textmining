import praw
import pandas as pd
import time
import datetime

CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGENT = ''

def getComments(folderLoc, wsbComments, subreddit):
    comments = []
    for s in subreddit.comments(limit=10000):
        #if (s.is_video == False):
            d = {'author': s.author, 'created_utc': s.created_utc, 'id': s.id, 'body': s.body, 'subreddit': s.subreddit,
                 'subreddit_id': s.subreddit, 'parent_id':s.parent_id, 'link_id':s.link_id}
            comments.append(d)

    df = pd.DataFrame(comments)
    print(df)

    df.to_csv(folderLoc + wsbComments, index=False)

def getPosts(folderLoc, wsbPosts, subreddit):
    posts = []
    # for s in subreddit.top(limit=256):
    for s in subreddit.new(limit=1000):
        if (s.is_video == False):
            d = {'author': s.author, 'author_premium': '', 'created_utc': s.created_utc,
                 'domain': s.domain,
                 'id': s.id, 'num_comments': s.num_comments, 'selftext': s.selftext, 'subreddit': s.subreddit,
                 'subreddit_id': s.subreddit,
                 'title': s.title, 'upvote_ratio': s.upvote_ratio}
            posts.append(d)

    df = pd.DataFrame(posts)
    print(df)

    df.to_csv(folderLoc + wsbPosts, index=False)

# Pulls from the reddit API with PRAW and writes to the CSV.
#TODO: add comment extraction.
if __name__ == '__main__':
    currentTime = time.strftime('%Y%m%d%H%M%S', time.localtime())
    print(currentTime)

    folderLoc = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\rawdata\\'
    wsbPosts = 'wallstreetbets_posts_' + currentTime + '.csv'
    wsbComments = 'wallstreetbets_comments_' + currentTime + '.csv'

    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    subreddit = reddit.subreddit('wallstreetbets')
    print(subreddit)

    getPosts(folderLoc, wsbPosts, subreddit)
    getComments(folderLoc, wsbComments, subreddit)
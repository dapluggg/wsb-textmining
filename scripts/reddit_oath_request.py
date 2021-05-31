import praw
import pandas as pd

CLIENT_ID = ''
CLIENT_SECRET = ''
USER_AGEN = ''

# Pulls from the reddit API with PRAW and writes to the CSV.
#TODO: add comment extraction.
if __name__ == '__main__':
    folderLoc = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\rawdata\\'
    wsbPosts = 'wallstreetbets_posts_live.tsv'
    wsbComments = 'wallstreetbets_comments_live.tsv'

    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    subreddit = reddit.subreddit('wallstreetbets')
    print(subreddit)

    posts = []
    for s in subreddit.new(limit=256):
        if (s.is_video == False):
            d = {'author': s.author,'author_premium':s.author_premium,'created_utc':s.created_utc,'domain':s.domain,
                 'id':s.id,'num_comments':s.num_comments,'body':s.selftext,'subreddit':s.subreddit,'subreddit_id':s.subreddit,
                 'title':s.title,'upvote_ratio':s.upvote_ratio}
            posts.append(d)

    df = pd.DataFrame(posts)
    print(df)

    df.to_csv(folderLoc + wsbPosts, sep='\t', index=False)
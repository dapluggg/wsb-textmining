#%%
import pandas as pd 
import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
MIN_DF=0.0002
MAX_DF = 0.001
import seaborn as sns
import re
import emoji
from tqdm import tqdm
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import model_selection
import pickle
from sklearn.model_selection import train_test_split




#%%
postsdf = pd.read_csv('/Users/shashanknagaraja/Library/Mobile Documents/com~apple~CloudDocs/Syracuse/Finished/IST 652 Scripting/wallstbets-sent/wsb-sent/final_output/reddit_wsb.csv')
#postsdf = postsdf.head(n=250000*2)

#%%
def vectorize_reviews_sklearn_tfidf(reviewsdf):
    sklearn_vec = TfidfVectorizer(encoding='latin-1', use_idf=True, stop_words='english', min_df=MIN_DF, max_df=MAX_DF)
    vecs = sklearn_vec.fit_transform(reviewsdf['title'].values.astype('U'))
    #print(sklearn_vec.vocabulary_)
    result = pd.DataFrame(vecs.toarray())
    result.columns = sklearn_vec.get_feature_names()
    return result
#%%
tfidf_vecs = vectorize_reviews_sklearn_tfidf(postsdf)
#%%
tfidf_vecs
# %%
def run_vader_analysis(tweets):
    '''
    Use VADER to get sentiment.
    '''
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for tweet in tqdm(tweets):
        vader_sentiment = analyzer.polarity_scores(tweet)
        sentiments.append(vader_sentiment)
        # print(f'{tweet[:50]}........{vader_sentiment}')
    return sentiments
vader_sent = run_vader_analysis(postsdf['title'])
#%%
sents = []
for score in vader_sent:
    if score['compound'] > 0:
        sents.append('positive')
    elif score['compound'] < 0:
        sents.append('negative')
    else:
        sents.append('neutral')
# %%
def train_models(X_train, y_train, X_test, y_test):
    dfs = []
    models = [
        ('MNB', MultinomialNB()),
        ('RF', RandomForestClassifier()),
        ('SVM-LIN', SVC(kernel='linear')),
        ('DT', DecisionTreeClassifier())
    ]

    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['positive', 'negative', 'neutral']

    trained_models = []
    for name, model in tqdm(models):
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=999)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred, target_names=target_names))
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        trained_models.append(model)
    final = pd.concat(dfs, ignore_index=True)

    return final

X_train, X_test, y_train, y_test = train_test_split(tfidf_vecs, sents, test_size=0.33, random_state=42)

# %%
result = train_models(X_train, y_train, X_test, y_test)
# %%

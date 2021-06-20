import pandas as pd
import pandasql as pdsql
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob as g
import collections as c

# SMOTE for oversampling.
from imblearn.over_sampling import SVMSMOTE, SMOTE

# Sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import plot_confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC

# NLTK
from nltk import pos_tag

# Read the data from parquet
def getData (filePath, nrows):
    glDf = pd.read_csv(filePath, nrows=nrows)
    return glDf


def plotTopUsedWords(inDf, threshold, label):
    # What are the most used words in the restaurant reviews?
    print('++++++++++ Plot token distribution by sentiment.')
    df = inDf.drop(['LABEL'], axis=1)
    df = df.sum()
    df = df.reset_index()
    df.columns = ['token', 'frequency']

    df = pdsql.sqldf('select token, sum(frequency) as freq_sum from df group by token order by freq_sum desc')
    df = df[:threshold]
    df['token_pos'] = pos_tag(df['token'])

    poss = []
    for x in df['token_pos']:
        t, p = zip(x)
        poss.append(p[0])
    df['pos_tag'] = poss

    ax = sns.barplot(x="token_pos", y="freq_sum", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Token')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Top ' + str(threshold) + ' Most Used ' + label + ' Words')
    plt.show()

    return df

# Build a COUNT vectorized data frame with sentiment and category labels.
# Uses English stop word list.
def buildCountVectorizedDf(content, labels):
    cv = CountVectorizer(input="content", stop_words='english')
    return buildVectorizedDf(content, labels, cv)

# Build a vectorized data frame from the passed vectorizer.
def buildVectorizedDf(content, labels, vectorizer):
    # Now lets vectorize the corpus.
    cvResults = vectorizer.fit_transform(content)
    cols = vectorizer.get_feature_names()

    # put to pandas
    df = pd.DataFrame(cvResults.toarray(), columns=cols)
    df['LABEL'] = labels
    print(df['LABEL'])

    return df

# %%
#filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210506\\wsb_post_results_*.csv'
filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210506\\wsb_*.csv'
files = g.glob(filePath, recursive=True)
print(files)
thisshitisfyaDf = pd.DataFrame()
drew1027Df = pd.DataFrame()
drew213Df = pd.DataFrame()
for f in files:
    print('processing file:', f)
    df = getData(f, 90000000)
    tsif = df[df['author'] == 'thisshitisfiya']
    d1027 = df[df['author'] == 'drew1027']
    d213 = df[df['author'] == 'drew213']
    thisshitisfyaDf = thisshitisfyaDf.append(tsif[['author','body']], ignore_index = True)
    drew1027Df = drew1027Df.append(d1027[['author', 'body']], ignore_index=True)
    drew213Df = drew213Df.append(d213[['author', 'body']], ignore_index=True)
print(thisshitisfyaDf)
print(thisshitisfyaDf.shape)
print(drew1027Df.shape)
print(drew213Df.shape)

tsifBodyArray = thisshitisfyaDf['body'].to_numpy()
tsifLabelArray = thisshitisfyaDf['author'].to_numpy()
tsifCntVecRes = buildVectorizedDf()

d1027BodyArray = thisshitisfyaDf['body'].to_numpy()
d1027LabelArray = thisshitisfyaDf['author'].to_numpy()

d213BodyArray = thisshitisfyaDf['body'].to_numpy()
d213LabelArray = thisshitisfyaDf['author'].to_numpy()


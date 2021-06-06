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


# Build a TFIDF vectorized data frame with sentiment and category labels.
# Uses English stop word list.
# Use unigram and bigram.
def buildTFIDFVectorizedDf(content, labels):
    cv = TfidfVectorizer(input="content", stop_words='english')
    return buildVectorizedDf(content, labels, cv)


# Build a COUNT vectorized data frame with sentiment and category labels.
# Uses English stop word list.
def buildCountVectorizedDf(content, labels):
    cv = CountVectorizer(input="content", stop_words='english')
    return buildVectorizedDf(content, labels, cv)


# Build a BINARY vectorized data frame with sentiment and category labels.
# Uses English stop word list.
def buildBinaryVectorizedDf(content, labels):
    cv = CountVectorizer(input="content", stop_words='english', binary=True)
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

# K-Fold cross validation.
# job name, data frame, number of folds, sklearn ml, test size, random state.
def kFoldCrossValidation_updown(name, df, folds, ml, size=0.33, rs=None):
    # models = []
    count = 1  # Count the iterations
    scores = pd.DataFrame()  # all scores added together for average.

    # Create test and train sets.  Iterate through the shuffled sets.
    rs = ShuffleSplit(n_splits=folds, test_size=size, random_state=rs)
    for train_idx, test_idx in rs.split(df):
        # print(train_idx, test_idx)

        train_o = df.iloc[train_idx]  # Get train by index.
        train_lable = train_o['LABEL']  # Seperate the labels
        train_o = train_o.drop(columns=['LABEL'])
        print(train_o.shape)
        print(train_lable.shape)


        test_o = df.iloc[test_idx]  # get test by index.
        test_lable = test_o['LABEL']  # Seperate the labels
        test_reviews = test_o.drop(columns=['LABEL'])  # post text
        print(test_o.shape)
        print(test_lable.shape)
        print(test_reviews.shape)

        counter = c.Counter(train_lable)
        print("Original sample counts", counter)

        # SMOTE
        # Using SMOTE to oversample the minority and under sample the majority.
        # Original sample from data:  0:1867, 1:260
        # strategy = {0: 775, 1: 795}
        print("SMOTE+++++")
        strategy='all'
        over = SMOTE(sampling_strategy=strategy)
        train_o_smote, train_lable_smote = over.fit_resample(train_o, train_lable)
        print(train_o_smote.shape)
        print(train_lable_smote.shape)

        # Validate the change by SMOTE.
        counter = c.Counter(train_lable_smote)
        print("Oversampled counts", counter)

        train_reviews = train_o_smote  # post text
        train_lable = train_lable_smote  # labels

        # Fit to the passed model.
        ml.fit(train_reviews, train_lable)

        # Now lets predict with the test set.
        predictions = ml.predict(test_reviews)

        # Score
        medianPredScore = ml.score(test_reviews, test_lable)
        prScores = precision_recall_fscore_support(test_lable, predictions, average='macro')

        # Create the confusion matrix.
        # cfm = confusion_matrix(test_lable, predictions)

        disp = plot_confusion_matrix(ml, test_reviews, test_lable)
        disp.ax_.set_title(name + ': KFold-' + str(count) + " Confusion Matrix")
        plt.show()

        scores = scores.append({'fold': count, 'precision': prScores[0],
                                'recall': prScores[1], 'f1': prScores[2],
                                'score': medianPredScore, 'mdl': ml}, ignore_index=True)

        scores['pr_diff'] = scores['precision'] - scores['recall']

        count = count + 1
    return scores

# Set RSI momentum flags.
def setRsiSignal(row):
    if row['rsi'] <= 30:
        val = 0 # 'OVER_SOLD'
    elif row['rsi'] >= 70:
        val = 4 # 'OVER_BOUGHT'
    elif (row['rsi'] < 50 and row['rsi'] > 30):
        val = 1 # 'BEARISH_MOMENTUM'
    elif (row['rsi'] >= 50 and row['rsi'] < 70):
        val = 3 # 'BULLISH_MOMENTUM'
    else:
        val = 2 #'NEUTRAL_MOMENTUM'
    return val
# %%
moderators = ['OPINION_IS_UNPOPULAR','CHAINSAW_VASECTOMY','WallStreetBot','bawse1','zjz','VisualMod','premier_',
              'notmikjaash','WaterCups69','XvGTM17','theycallme1','JohnnyCupcakes','Plechazunga_','HellzAngelz','Stylux',
              'TheDrallen','ClassicRust','rocketfuelandcoffee','RapsAboutDiablo','sdevil713','ThetaGang_wsb','The_Three_Nuts',
              'VacationLover1','FannyPackPhantom','CallsOnAlcoholism','Grumpy-james','GoBeaversOSU','WilliamNyeTho',
              'richtofin115','umbrellacorpbailout','Darkbyte','Pusherman_','teddy_riesling','TheIceCreamMansBro2',
              'Dan_inKuwait','DisabledSexRobot','onelot','SignedUpWhilePooping','Swedish_Chef_Bork_x3','GasolinePizza',
              'cafenegroporfa','Epidemilk','Memetron9000']
cols = ['author','author_premium','created_utc','domain','id','num_comments','body','subreddit','subreddit_id','title',
        'upvote_ratio','body_filtered','title_filtered','created_utc_datetime','doc_type','body_polarity',
        'body_subjectivity','body_vadar_sentiment','body_tickers','date','rsi','open','high','low','close','volume',
        'adjusted','ticker','title_vadar_sentiment','title_tickers']

#postPath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210506\\wsb_post_results_1.csv'

#filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210506\\wsb_post_results_*.csv'
filePath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\SVM_Train\\wsb_*.csv'
files = g.glob(filePath, recursive=True)
print(files)
postDf = pd.DataFrame()
for f in files:
    print('processing file:', f)
    df = getData(f, 90000000)
    df = df[df['body_tickers'] == 'GME']
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df = df[(df.close > 0) & (df.open > 0)]
    df['gain'] = df.close - df.open
    df['up_down'] = np.where(df.gain > 0, 1, 0)  #UP = 1 DOWN = 0
    df['rsi_signal'] = df.apply(setRsiSignal, axis=1)
    df['created_utc_date'] = pd.to_datetime(df['created_utc_datetime'], format='%Y-%m-%d')
    df = df.sort_values(by=['created_utc_date'])
    #df = df[(df.created_utc_date >= '2021-01-05 00:00:00') & (df.created_utc_date <= '2021-01-22 00:00:00') ]
    df = df[~df.author.isin(moderators)]  # Remove moderators
    postDf = postDf.append(df, ignore_index = True)
print(postDf[['created_utc_date','author','gain','up_down','rsi_signal']])
print(postDf.shape)

# Sentiment distribution
dd = pdsql.sqldf('select up_down, count(up_down) up_down_frequency from postDf group by up_down order by up_down_frequency desc')
plt.figure(figsize=(10, 10))
ax = sns.barplot(x="up_down", y="up_down_frequency", data=dd)
plt.show()

dd = pdsql.sqldf('select rsi_signal, count(rsi_signal) rsi_signal_frequency from postDf group by rsi_signal order by rsi_signal_frequency desc')
plt.figure(figsize=(10, 10))
ax = sns.barplot(x="rsi_signal", y="rsi_signal_frequency", data=dd)
plt.show()

dd = pdsql.sqldf('select rsi_signal, up_down, count(rsi_signal) frequency from postDf group by rsi_signal, up_down order by frequency desc')
plt.figure(figsize=(10, 10))
ax = sns.barplot(x="rsi_signal", y="frequency", hue='up_down', data=dd)
plt.show()




print ('===== UP DOWN PREDICTIONS =====')

# Extract from the datafame the information needed to build the
# count vectorizer.
rArray = postDf['body'].to_numpy()
lArray = postDf['up_down'].to_numpy()

# Vectorize with the sentiment label.
print("TFIDF SVM >- Cross Validation Results.")
ud_tfidf_vector = buildTFIDFVectorizedDf(rArray, lArray)
print(ud_tfidf_vector)
scores = kFoldCrossValidation_updown("SVM_TFIDF", ud_tfidf_vector, 1, SVC())
print("F1 Cross Val Score: ", round(scores['score'].mean(), 3), '\n')

# Get the best model off f1 score.
min_pr_diff = scores['f1'].max()
m = scores[scores['f1'] == min_pr_diff].reset_index()
print(m[['fold','f1','precision','recall','score','pr_diff']])

print("COUNT SVM >- Cross Validation Results.")
ud_count_vector = buildCountVectorizedDf(rArray, lArray)
print(ud_count_vector)
scores = kFoldCrossValidation_updown("SVM_COUNT", ud_count_vector, 1, SVC())
print("F1 Cross Val Score: ", round(scores['score'].mean(), 3), '\n')

# Get the best model off f1 score.
min_pr_diff = scores['f1'].max()
m = scores[scores['f1'] == min_pr_diff].reset_index()
print(m[['fold','f1','precision','recall','score','pr_diff']])

print ('===== RSI PREDICTIONS =====')

# refresh labels.
rArray = postDf['body'].to_numpy()
lArray = postDf['rsi_signal'].to_numpy()

# Vectorize with the sentiment label.
print("TFIDF SVM >- Cross Validation Results.")
ud_tfidf_vector = buildTFIDFVectorizedDf(rArray, lArray)
print(ud_tfidf_vector)
scores = kFoldCrossValidation_updown("SVM_TFIDF", ud_tfidf_vector, 1, SVC())
print("F1 Cross Val Score: ", round(scores['score'].mean(), 3), '\n')

# Get the best model off f1 score.
min_pr_diff = scores['f1'].max()
m = scores[scores['f1'] == min_pr_diff].reset_index()
print(m[['fold','f1','precision','recall','score','pr_diff']])

print("COUNT SVM >- Cross Validation Results.")
ud_count_vector = buildCountVectorizedDf(rArray, lArray)
print(ud_count_vector)
scores = kFoldCrossValidation_updown("SVM_COUNT", ud_count_vector, 1, SVC())
print("F1 Cross Val Score: ", round(scores['score'].mean(), 3), '\n')

# Get the best model off f1 score.
min_pr_diff = scores['f1'].max()
m = scores[scores['f1'] == min_pr_diff].reset_index()
print(m[['fold','f1','precision','recall','score','pr_diff']])
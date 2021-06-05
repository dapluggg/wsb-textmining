import pandas as pd
import pandasql as pdsql
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
def kFoldCrossValidation(name, df, folds, ml, size=0.33, rs=None):
    # models = []
    count = 1  # Count the iterations
    scores = pd.DataFrame()  # all scores added together for average.

    # Create test and train sets.  Iterate through the shuffled sets.
    rs = ShuffleSplit(n_splits=folds, test_size=size, random_state=rs)
    for train_idx, test_idx in rs.split(df):
        # print(train_idx, test_idx)

        train_o = df.iloc[train_idx]  # Get train by index.
        # print(train_o.shape)

        test_o = df.iloc[test_idx]  # get test by index.
        # print(test_o.shape)

        train_lable = train_o['LABEL']  # Seperate the labels
        train_reviews = train_o.drop(columns=['LABEL'])  # Reviews text

        test_lable = test_o['LABEL']  # Seperate the labels
        test_reviews = test_o.drop(columns=['LABEL'])  # Reviews text

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
        val = 'OVER_SOLD'
    elif row['rsi'] >= 70:
        val = 'OVER_BOUGHT'
    elif row['rsi'] <= 48:
        val = 'BEARISH_MOMENTUM'
    elif row['rsi'] >= 52:
        val = 'BULLISH_MOMENTUM'
    else:
        val = 'NEUTRAL_MOMENTUM'
    return val
# %%
moderators = ['OPINION_IS_UNPOPULAR','CHAINSAW_VASECTOMY','WallStreetBot','bawse1','zjz','VisualMod','premier_',
              'notmikjaash','WaterCups69','XvGTM17','theycallme1','JohnnyCupcakes','Plechazunga_','HellzAngelz','Stylux',
              'TheDrallen','ClassicRust','rocketfuelandcoffee','RapsAboutDiablo','sdevil713','ThetaGang_wsb','The_Three_Nuts',
              'VacationLover1','FannyPackPhantom','CallsOnAlcoholism','Grumpy-james','GoBeaversOSU','WilliamNyeTho',
              'richtofin115','umbrellacorpbailout','Darkbyte','Pusherman_','teddy_riesling','TheIceCreamMansBro2',
              'Dan_inKuwait','DisabledSexRobot','onelot','SignedUpWhilePooping','Swedish_Chef_Bork_x3','GasolinePizza',
              'cafenegroporfa','Epidemilk','Memetron9000',]

postPath = 'C:\\Users\\green\\Documents\\Syracuse_University\\IST_736\\Project\\wsb-textmining\\processed\\20210506\\wsb_post_results_1.csv'
postDf = getData(postPath, 90000000)
postDf = postDf[postDf['body_tickers'] == 'GME']
postDf['close'] = postDf['close'].astype(float)
postDf['open'] = postDf['open'].astype(float)
postDf = postDf[(postDf.close > 0) & (postDf.open > 0)]
postDf['gain'] = postDf.close - postDf.open
postDf['up_down'] = np.where(postDf.gain > 0, 'UP', 'DOWN')
postDf['rsi_signal'] = postDf.apply(setRsiSignal, axis=1)
postDf['created_utc_date'] = pd.to_datetime(postDf['created_utc_datetime'], format='%Y-%m-%d')
postDf = postDf.sort_values(by=['created_utc_date'])
# Remove moderators
postDf = postDf[~postDf.author.isin(moderators)]
print(postDf[['created_utc_date','author','gain','up_down','rsi_signal']])

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

# Extract from the datafame the information needed to build the
# count vectorizer.
rArray = postDf['body'].to_numpy()
sentArray = postDf['up_down'].to_numpy()

# Vectorize with the sentiment label.
print("TFIDF LINEAR SVM >- Cross Validation Results.")
ud_tfidf_vector = buildTFIDFVectorizedDf(rArray, sentArray)
print(ud_tfidf_vector)
scores = kFoldCrossValidation("SVM_TFIDF", ud_tfidf_vector, 5, SVC(kernel='linear'))
print("F1 Cross Val Score: ", round(scores['score'].mean(), 3), '\n')

# Get the best model off f1 score.
min_pr_diff = scores['f1'].max()
m = scores[scores['f1'] == min_pr_diff].reset_index()
print(m[['fold','f1','precision','recall','score','pr_diff']])
#%%
from main import DATA_DIR_LOCAL
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from tqdm import tqdm
import pickle
models_filename = DATA_DIR_LOCAL / 'output' / 'models.pkl'

def train_models(X_train, y_train, X_test, y_test):
    dfs = []
    models = [
        ('MNB', MultinomialNB()),
        ('RF', RandomForestClassifier()),
        #('KNN', KNeighborsClassifier()),
        ('SVM-LIN', SVC(kernel='linear')),
        ('SVM-POL', SVC(kernel='poly')),
        ('SVM-RBF', SVC(kernel='rbf')),
        ('GNB', GaussianNB())
    ]

    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['Democrat', 'Republican']

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
    pickle.dump(trained_models, open(models_filename, 'wb'))

    return final


    
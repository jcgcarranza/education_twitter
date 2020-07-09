# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020

@author: Juan Carlos Gomez

This code conduct a classification of users based on their education
level.

It uses the tweets posted by users to train and test classification models.

It considers the following levels: superior (1) or no superior (0)

It uses a set of files containing different features extracte from the tweets:
    words, emoticons, links, ats, hashtags and abreviations

As models, it uses support vector machines, multinomial naive bayes or decision trees.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
import numpy as np

def my_tokenizer(text):
    """Split a text using spaces as separator.

    Keyword arguments:
    s -- string containing a text formed by fetures separated by spaces.

    Output:
    a list of features (each feature is a string)
    """
    return text.split()

def read_data(file):
    """Read data from a file line by line, removing the new line chars at the
    end of the line, and store the lines in a list.

    Keyword arguments:
    file -- string with the file name to read.

    Output:
    a list with the lines of the file as strings.
    """
    data = []
    with open(file, 'r', encoding='utf-8') as reader:
        for line in reader:
            line = line.strip()
            data.append(line)
    return data

def group_per_user(corpus_o, labels_o, users_o):
    """Group a set of tweets (strings) from a corpus per user.

    Keyword arguments:
    corpus -- a list of tweets (strings) represented as a set of features
            separated by spaces.
    labels -- a list of labels for the category of each tweet
    users -- a list of users to which each tweet belongs.

    Ouput:
    two lists of tweets and labels grouped per user.
    """
    corpus_grouped = []
    labels_grouped = []
    d_text = {}
    d_label = {}
    for user, label, text in zip(users_o, labels_o, corpus_o):
        d_text[user] = d_text.get(user, '')+text+' '
        d_label[user] = label
    for user in d_text:
        corpus_grouped.append(d_text[user])
        labels_grouped.append(d_label[user])
    return corpus_grouped, labels_grouped


#Definition of configuration parameters
W_D = 'D:/Documentos/codeanddata/datasets/2019_twitter_education_degree_v2/'
LABELS_FILE = W_D+'education.txt'
USERS_FILE = W_D+'users.txt'
SW_FILE = W_D+'spanish_stopwords.txt'
FEAT = 6 #Number of feature to read: 0 to 6
MODEL = 'dt' #Model to build: svm, mnb or dt
WEIGHT = 'tfidf' #How to weight the vectors: tfidf, bin (binary, only for words: feature 0)

#Features file names
FEAT_FILES = ['split/words.txt', 'split/emoticons.txt', 'split/hashtags.txt',
              'split/ats.txt', 'split/links.txt', 'split/abvs.txt',
              'split/all.txt']

#Load labels and user data
print('Loading users labels...')
labels = read_data(LABELS_FILE)
print('Users labels loaded!')
print('Loading user list...')
users = read_data(USERS_FILE)
print('User list loaded!')


#Load tweet features
print('Loading tweet features...')
corpus = read_data(W_D+FEAT_FILES[FEAT])
print('Tweet features loaded!')

#Group per user
print('Grouping tweet data per user...')
corpus, labels = group_per_user(corpus, labels, users)
print('Tweet data grouped!')
labels = np.asarray([int(l) for l in labels])


#Create 10-folds for experiments
skf = StratifiedKFold(n_splits=10)
f1s = []
aucs = []
i = 0 #Fold counter
print('Building models per fold...')
for train_index, test_index in skf.split(corpus, labels):
    print('Fold :', i)
    data_train = [corpus[x] for x in train_index]
    data_test = [corpus[x] for x in test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]
    if WEIGHT == 'tfidf':
        vectorizer = TfidfVectorizer(norm='l2', analyzer='word', tokenizer=my_tokenizer)
    elif WEIGHT == 'bin':
        vectorizer = CountVectorizer(analyzer='word', tokenizer=my_tokenizer,
                                     binary=True) #Binary values
    else:
        vectorizer = CountVectorizer(analyzer='word',
                                     tokenizer=my_tokenizer) #Frequency values

    vec_train = vectorizer.fit_transform(data_train)

    if MODEL == 'svm':
        #Optimize for C
        cs = [0.1, 1.0, 10.0, 100.0] #Value of C for optimization in SVM
        BEST_C = 0
        BEST_SCORE = 0
        for c in cs:
            clf_inner = svm.LinearSVC(C=c, max_iter=15000)
            sub_skf = StratifiedKFold(n_splits=3)
            scores_inner = cross_val_score(clf_inner, vec_train, labels_train,
                                           scoring='f1_macro', cv=sub_skf)
            score = np.mean(scores_inner)
            if score > BEST_SCORE:
                BEST_SCORE = score
                BEST_C = c
        #Define the SVM final model with the best C
        clf = svm.LinearSVC(C=BEST_C, max_iter=15000)
    elif MODEL == 'mnb':
        #Define a model for Naive Bayes
        clf = MultinomialNB()
    else:
        #Define a model for Decision Trees
        clf = DecisionTreeClassifier(criterion='entropy')
    #Create the final model
    clf.fit(vec_train, labels_train)

    #Test the final model with the test fold and compute metrics
    vec_test = vectorizer.transform(data_test)
    predicted = clf.predict(vec_test)
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    auc = metrics.roc_auc_score(labels_test, predicted)
    f1s.append(f1_macro)
    aucs.append(auc)
    i += 1
print('Models finished!\n')

#Print averaged results and standard deviations
print('Results with feature', FEAT, ', model', MODEL, 'and weight', WEIGHT, '\n')

print('Education F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s)))
print('Education AUC: %0.2f (+/- %0.2f)' % (np.mean(aucs), np.std(aucs)))
    
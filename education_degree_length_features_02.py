# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020

@author: Juan Carlos Gomez

This code conduct a classification of users based on their education level.
It uses the tweets posted by users to train and test classification models.

It considers the following levels: superior (1) or no superior (0)

It uses as features the average tweet length per user, considering the number
of words, emoticons, links, ats, hashtags and abreviations.

As models, it uses support vector machines, multinomial naive bayes or decision trees.
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
import numpy as np


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

def read_counts(file):
    """Read data from a file line by line, split each line by spaces and
    counts the number of resulting tokens.

    Keyword arguments:
    file -- string with the file name to read.

    Output:
    a list with the number of tokens in each line.
    """
    data = []
    with open(file, 'r', encoding='utf-8') as reader:
        for line in reader:
            tokens = line.strip().split()
            data.append(len(tokens))
    return data

def group_per_user(corpus_o, labels_o, users_o):
    """Averages a set of counts of tokens of tweets per user.

    Keyword arguments:
    corpus -- a list of number of tokens in each tweet.
    labels -- a list of labels for the category of each tweet
    users -- a list of users to which each tweet belongs.

    Ouput:
    two lists, one with the average lengths (number of tokens) of the tweets
    from each user, and another with the label of each user.
    """
    corpus_grouped = []
    labels_grouped = []
    d_count = {}
    d_label = {}
    d_n = {}
    for user, label, count in zip(users_o, labels_o, corpus_o):
        d_count[user] = d_count.get(user, 0)+count
        d_n[user] = d_n.get(user, 0)+1
        d_label[user] = label
    for user in d_count:
        corpus_grouped.append(d_count[user]/d_n[user])
        labels_grouped.append(d_label[user])
    return corpus_grouped, labels_grouped

def form_vectors(words_o, emojis_o, hashs_o, ats_o, links_o, abvs_o, alls_o):
    """Form a vector per tweet by grouping the averages of the set of features.

    Keyword arguments:
    corpus -- a list of number of tokens in each tweet.
    labels -- a list of labels for the category of each tweet
    users -- a list of users to which each tweet belongs.

    Ouput:
    two lists, one with the average lengths of the tweets from each user, and
    another with the label of each user.
    """
    corpus_d = [[word, emoji, has, at, link, abv, al] for word, emoji, has, at,
                link, abv, al in zip(words_o, emojis_o, hashs_o, ats_o,
                                     links_o, abvs_o, alls_o)]
    return np.array(corpus_d)

#Definition of configuration parameters
W_D = 'D:/Documentos/codeanddata/datasets/2019_twitter_education_degree_v2/'
LABELS_FILE = W_D+'education.txt'
USERS_FILE = W_D+'users.txt'
GLOVE_FILE = W_D+'glove-sbwc.i25.vec'
WORDS_FILE = W_D+'split/words.txt'
MODEL = 'dt' #Model to build: svm, mnb or dt
NORM = True #Normalize the feature vectors: True or False

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
words = read_counts(W_D+FEAT_FILES[0])
emojis = read_counts(W_D+FEAT_FILES[1])
hashs = read_counts(W_D+FEAT_FILES[2])
ats = read_counts(W_D+FEAT_FILES[3])
links = read_counts(W_D+FEAT_FILES[4])
abvs = read_counts(W_D+FEAT_FILES[5])
alls = read_counts(W_D+FEAT_FILES[6])
print('Tweet features loaded!')

#Group per user
print('Grouping tweet data per user...')
words, labels_w = group_per_user(words, labels, users)
emojis, labels_e = group_per_user(emojis, labels, users)
hashs, labels_h = group_per_user(hashs, labels, users)
ats, labels_at = group_per_user(ats, labels, users)
links, labels_l = group_per_user(links, labels, users)
abvs, labels_ab = group_per_user(abvs, labels, users)
alls, labels = group_per_user(alls, labels, users)
print('Tweet data grouped!')
labels = np.asarray([int(l) for l in labels])


#Form feature vectors
print('Forming feature vectors...')
corpus = form_vectors(words, emojis, hashs, ats, links, abvs, alls)
print('Vectors formed!')
if NORM:
    corpus = normalize(corpus)

#Create 10-folds for experiments
skf = StratifiedKFold(n_splits=10)
f1s = []
aucs = []
i = 0 #Fold counter
print('Building models per fold...')
for index_train, index_test in skf.split(corpus, labels):
    print('Fold :', i)
    vec_train = corpus[index_train]
    labels_train = labels[index_train]
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
    vec_test = corpus[index_test]
    labels_test = labels[index_test]
    predicted = clf.predict(vec_test)
    f1_macro = metrics.f1_score(labels_test, predicted, average='macro')
    auc = metrics.roc_auc_score(labels_test, predicted)
    f1s.append(f1_macro)
    aucs.append(auc)
    i += 1
print('Models finished!\n')

#Print averaged results and standard deviations
print('Results with average length of tweets, and ', MODEL, ', model', '\n')

print('Education F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print('Education AUC: %0.2f (+/- %0.2f)' % (np.mean(aucs), np.std(aucs) * 2))

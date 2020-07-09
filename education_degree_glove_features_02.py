# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020

@author: Juan Carlos Gomez

This code conduct a classification of users based on their education level.
It uses the tweets posted by users to train and test classification models.

It considers the following levels: superior (1) or no superior (0)

It uses as features GloVe word vectors computed from the words of the tweets.

As models, it uses support vector machines or decision trees.
"""

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
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

def read_glove(file):
    """ Read a file containing word vectors. The file must contain a word
    vector per line, with the elements of the vector separated by spaces. The
    first element is the word and the remaining are float numbers representing
    the word vector.

    The first line must contain the number of words in the file and the legth
    of the word vectors.

    Keyword arguments:
    file -- string with the file name to read.

    Output:
    a dictionary of the word vectors, the key is the word and the value is a
    list representing the vector for such word.
    """
    d_g = {}
    with open(file, 'r', encoding='utf-8') as reader:
        reader.readline()
        for line in reader:
            tokens = line.strip().split()
            v_d = [float(token) for token in tokens[1:]]
            d_g[tokens[0]] = v_d
    return d_g

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
GLOVE_FILE = W_D+'glove-sbwc.i25.vec'
WORDS_FILE = W_D+'split/words.txt'
MODEL = 'dt' #Model to build: svm or dt

#Load GloVe vectors
print('Loading GloVe vectors...')
glove_dict = read_glove(GLOVE_FILE)
print('GloVe vectors loaded!')

#Load labels and user data
print('Loading users labels...')
labels = read_data(LABELS_FILE)
print('Users labels loaded!')
print('Loading user list...')
users = read_data(USERS_FILE)
print('User list loaded!')

#Load tweet features
print('Loading tweet features...')
corpus = read_data(WORDS_FILE)
print('Tweet features loaded!')

#Group per user
print('Grouping tweet data per user...')
corpus, labels = group_per_user(corpus, labels, users)
print('Tweet data grouped!')
labels = np.asarray([int(l) for l in labels])

k = next(iter(glove_dict)) #Get a key from the GloVe dictionary
N = len(glove_dict[k]) #Get the length of GloVe word vectors

#Split documents in words
corpus_words = [line.split() for line in corpus]
corpus = []

#Compute document vectors using word vectors
for document in corpus_words:
    vect = np.zeros((N))
    i = 0
    for word in document:
        if word in glove_dict:
            vect += glove_dict[word]
            i += 1
    vect = vect/i
    corpus.append(vect)
corpus = np.array(corpus)

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
print('Results with GloVe vectors, and ', MODEL, ', model', '\n')

print('Education F1: %0.2f (+/- %0.2f)' % (np.mean(f1s), np.std(f1s) * 2))
print('Education AUC: %0.2f (+/- %0.2f)' % (np.mean(aucs), np.std(aucs) * 2))

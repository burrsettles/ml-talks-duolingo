"""
Burr Settles
Duolingo ML Dev Talk #4: Intro to Recomender Systems

Simple linear regression-based movie rating predictions with scikit-learn.
"""

import argparse
import csv
import sys
import random
import math

from collections import defaultdict

from scipy.stats.stats import pearsonr

import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.feature_extraction import DictVectorizer

def addtags(movie):
    taglist = MOVIES[movie['movie']][1]
    movie.update(dict((t, 1.) for t in taglist))
    if 'u' in args.mode:
        user = movie['user']
        movie.update(dict((t+'_'+user, 1.) for t in taglist))


MOVIES = {}

TRAIN_X = []
TRAIN_Y = []
TEST_X = []
TEST_Y = []

AVERAGE = 3.54360826

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', dest='mode', default='', action='store')

if __name__ == '__main__':

    args = parser.parse_args()

    # read in movie data
    with open('data/movies.csv', 'rbU') as mf:
        m_reader = csv.reader(mf, delimiter=',', quotechar='"')
        for row in m_reader:
            (movie, title, tags) = row
            MOVIES[movie] = (title, tags.lower().split('|'))

    # load the data set
    ct = 0
    with open('data/ratings.csv', 'rbU') as rf:
        r_reader = csv.reader(rf, delimiter=',', quotechar='|')
        for row in r_reader:
            try:
                (user, movie, rating, timestamp) = row
                rating = float(rating)
                inst = {'user': user, 'movie': movie}
                # stratify train/test set
                if ct % 10 == 0:
                    TEST_X.append(inst)
                    TEST_Y.append(rating)
                else:
                    TRAIN_X.append(inst)
                    TRAIN_Y.append(rating)
                ct += 1
            except:
                continue

    # add (optional) genre tag features
    if 'g' in args.mode:
        for inst in TRAIN_X:
            addtags(inst)
        for inst in TEST_X:
            addtags(inst)

    # train model
    y = np.array(TRAIN_Y)
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(TRAIN_X).toarray()
    model = linear_model.SGDRegressor(penalty='none')
    model.fit(X, y)
    model_weights = vectorizer.inverse_transform(model.coef_)[0]
    with open('lr.weights-%s.txt' % args.mode, 'w') as f:
        f.write('%s\t%f\n' % ('(intercept)', model.intercept_))
        f.writelines([('%.4f\t%s\n' % (float(v), str(k.encode("utf-8")))) for (k,v) in model_weights.items()])

    # predict test set ratings
    tX = vectorizer.transform(TEST_X).toarray()
    PREDS = model.predict(tX)

    # report mean absolute error (MAE) pearson correlation
    mae = sum(abs(TEST_Y[i] - PREDS[i]) for i in range(len(PREDS)))/len(PREDS)
    print 'MAE = ', round(mae, 3)
    r, p = pearsonr(PREDS, TEST_Y)
    print 'cor = ', round(r, 3)

"""
Burr Settles
Duolingo ML Dev Talk #4: Intro to Recomender Systems

Simple similarity-based movie rating predictions.
"""

import argparse
import csv
import sys
import random
import math

from collections import defaultdict

from scipy.stats.stats import pearsonr


# similarity cache
SIM_CACHE = {}

# movie data dictionary (title, etc.)
MOVIES = {}

# sparse vectors of user/movie ratings (and vice versa)
U_DICT = defaultdict(lambda : defaultdict(float))
M_DICT = defaultdict(lambda : defaultdict(float))

TESTSET = []

AVERAGE = 3.54360826

KEY_MOVIES = '260 2942 5618'.split()


def norm(m):
    return math.sqrt(sum(x**2 for x in M_DICT[m].values()))

def similarity(m1, m2):
    if m1 in SIM_CACHE and m2 in SIM_CACHE[m1]:
        return SIM_CACHE[m1][m2]
    elif m2 in SIM_CACHE and m1 in SIM_CACHE[m2]:
        return SIM_CACHE[m2][m1]
    else:
        try:
            sim = sum(v * M_DICT[m2].get(k, 0) for k, v in M_DICT[m1].iteritems())/(norm(m1) * norm(m2))
        except:
            sim = 0.
        SIM_CACHE.setdefault(m1, {}).update({m2: sim})
        return sim


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', dest='mode', default='gu', action='store')

if __name__ == '__main__':

    args = parser.parse_args()

    # read in movie data
    with open('data/movies.csv', 'rbU') as mf:
        m_reader = csv.reader(mf, delimiter=',', quotechar='"')
        for row in m_reader:
            (movie, title, tags) = row
            MOVIES[movie] = title
            # add (optional) genre tags to movie vector
            if 'g' in args.mode:
                taglist = tags.lower().split('|')
                M_DICT[movie].update(dict((t, 1.) for t in taglist))
            else:
                M_DICT[movie].update({})

    # load the data set
    ct = 0
    AVERAGE = 0.
    with open('data/ratings.csv', 'rbU') as rf:
        r_reader = csv.reader(rf, delimiter=',', quotechar='|')
        for row in r_reader:
            try:
                (user, movie, rating, timestamp) = row
                rating = float(rating)
                AVERAGE += rating
                # stratify train/test set
                if ct % 10 == 0:
                    TESTSET.append((user, movie, rating))
                else:
                    U_DICT[user].update({movie: rating})
                    if 'u' in args.mode:
                        M_DICT[movie].update({user: rating})
                ct += 1
            except:
                continue
    AVERAGE /= ct

    print 'number of users:\t%d' % len(U_DICT)
    print 'number of movies:\t%d' % len(M_DICT)
    print 'testset size:', len(TESTSET)
    print 'average rating:', AVERAGE

    # predict ratings for test set
    PREDS = []
    for (user, movie, rating) in TESTSET:
        pred = 0. + sum(similarity(movie, m) * U_DICT[user][m] for m in U_DICT[user])
        if pred > 0.:
            pred /= sum(similarity(movie, m) for m in U_DICT[user])
        else:
            pred = AVERAGE
        PREDS.append(pred)

    # report mean absolute error (MAE) and pearson correlation
    mae = sum(abs(TESTSET[i][2] - PREDS[i]) for i in range(len(PREDS)))/len(PREDS)
    print 'MAE = ', round(mae, 3)
    r, p = pearsonr(PREDS, [x[2] for x in TESTSET])
    print 'cor = ', round(r, 3)

    # report MAE for fixed prediction
    mae = sum(abs(TESTSET[i][2] - AVERAGE) for i in range(len(PREDS)))/len(PREDS)
    print 'CONTROL = ', round(mae, 3)

    # find "nearest neighbors" for "key movies"...
    for k in KEY_MOVIES:
        print '-'*80
        k_sims = dict((m, similarity(k, m)) for m in M_DICT.keys() if m != k)
        best = sorted(k_sims.items(), key=lambda x: x[1], reverse=True)
        print MOVIES[k]
        for m, sim in best[:5]:
            print sim, MOVIES[m]






"""
Burr Settles
Duolingo ML Dev Talk #4: Intro to Recomender Systems

Alternating least squared (ALS) approach to matrix factorization, using stochastic gradient
descent (SGD). Loosely inspired by Simon Funk (http://sifter.org/~simon/journal/20061211.html)
"""

import argparse
import csv
import sys
import random
import math

from collections import defaultdict

from scipy.stats.stats import pearsonr

MOVIES = {}     # movie data dictionary (title, etc.)
TRAIN_SET = []
TEST_SET = []

ETA = .002      # learning rate
ALPHA = 25.     # for smoothing user/movie "bias" parameters
LAMBDA = 0.02   # L2 regularization weight

# optional "bias" parameters
AVERAGE = 0.
U_OFFSETS = defaultdict(float)
M_OFFSETS = defaultdict(float)

# low-rank matrices
U_DIMS = []
M_DIMS = []

# similarity cache
SIM_CACHE = {}

NUM_ITERATIONS = 250


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

def predict(user, movie):
    return AVERAGE + U_OFFSETS[user] + M_OFFSETS[movie] + \
        sum(U_DIMS[d][user]*M_DIMS[d][movie] for d in range(len(U_DIMS)))

def evaluate(DATA_SET):
    PREDS = [predict(u, m) for (u, m, r) in DATA_SET]
    REALS = [r for (u, m, r) in DATA_SET]
    mae = sum(abs(REALS[i] - PREDS[i]) for i in range(len(PREDS)))/len(PREDS)
    print 'MAE = ', round(mae, 3)
    r, p = pearsonr(PREDS, REALS)
    print 'cor = ', round(r, 3)


parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', dest='mode', default='ao', action='store')
parser.add_argument('-n', dest='num_dims', type=int, default=10, action='store')

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
        r_reader = csv.reader(rf, delimiter=',', quotechar='"')
        for row in r_reader:
            try:
                (user, movie, rating, timestamp) = row
                rating = float(rating)
                # stratify train/test set
                if ct % 10 == 0:
                    TEST_SET.append((user, movie, rating))
                else:
                    TRAIN_SET.append((user, movie, rating))
                ct += 1
            except:
                continue

    print 'train=%d' % len(TRAIN_SET)
    print 'test=%d' % len(TEST_SET)

    # if these flags are set, compute "bias" parameters
    if 'a' in args.mode:

        # compute the average rating
        AVERAGE = sum(x[2] for x in TRAIN_SET)/len(TRAIN_SET)
        print 'computing average: %.4f' % AVERAGE

        # compute user/movie "offsets"
        if 'o' in args.mode:
            UT = defaultdict(list)
            MT = defaultdict(list)
            for inst in TRAIN_SET:
                (user, movie, rating) = inst
                UT[user].append(rating)
                MT[movie].append(rating)
            for user in UT:
                U_OFFSETS[user] = (ALPHA*AVERAGE + sum(UT[user]))/(ALPHA + len(UT[user])) - AVERAGE
            for movie in MT:
                M_OFFSETS[movie] = (ALPHA*AVERAGE + sum(MT[movie]))/(ALPHA + len(MT[movie])) - AVERAGE
            print 'done computing offsets'

    print '-'*80
    print 'STARTING POINT'
    evaluate(TEST_SET)

    # SGD training loop: for each latent dimension...
    for d in range(args.num_dims):

        # initialize to small value
        U_DIMS.append(defaultdict(lambda: .1))
        M_DIMS.append(defaultdict(lambda: .1))

        # do multiple training passes
        for it in range(NUM_ITERATIONS):
            random.shuffle(TRAIN_SET)
            for inst in TRAIN_SET:
                (user, movie, rating) = inst
                err = rating - predict(user, movie)
                # alternate updating use or movie parameters
                if it % 2 == 0:
                    U_DIMS[d][user] +=  ETA * err * M_DIMS[d][movie]    # error update
                    U_DIMS[d][user] -=  ETA * LAMBDA * U_DIMS[d][user]  # L2 regularization
                else:
                    M_DIMS[d][movie] += ETA * err * U_DIMS[d][user]
                    M_DIMS[d][movie] -= ETA * LAMBDA * M_DIMS[d][movie]
        print '-'*80
        print 'DIMENSION', d
        evaluate(TEST_SET)

    # write out the movie dimensions
    with open('mf.movies-%s.txt' % args.mode, 'w') as mf:
        for movie in M_DIMS[0]:
            for d in range(len(M_DIMS)):
                mf.write('%.4f\t' % M_DIMS[d][movie])
            mf.write('%s\t%s\n' % (MOVIES[movie][0], '_'.join(MOVIES[movie][1])))

    # write out the user dimensions
    with open('mf.users-%s.txt' % args.mode, 'w') as uf:
        for user in U_DIMS[0]:
            for d in range(len(U_DIMS)):
                uf.write('%.4f\t' % U_DIMS[d][user])
            uf.write('%s\n' % user)

    # find "nearest neighbors" for "key movies" in the latent space...
    M_DICT = {}
    for movie in MOVIES:
        M_DICT[movie] = {d: M_DIMS[d][movie] for d in range(len(M_DIMS))}
    KEY_MOVIES = '260 2942 5618'.split()
    for k in KEY_MOVIES:
        print '%'*80
        print M_DICT[k]
        k_sims = dict((m, similarity(k, m)) for m in M_DICT.keys() if m != k)
        best = sorted(k_sims.items(), key=lambda x: x[1], reverse=True)
        print MOVIES[k]
        for m, sim in best[:5]:
            print sim, MOVIES[m]


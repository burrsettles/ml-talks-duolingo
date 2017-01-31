"""
Burr Settles
Duolingo ML Dev Talk #3: Clustering

EM-GMM (expectaction maximization with Gaussian mixture models) clustering example using
scikit-learn.
"""

import argparse
import math
import json

import numpy as np

from bs4 import BeautifulSoup
from sklearn.mixture import GaussianMixture

# cluster colors (for map visualizations, up to 8)
COLORS = '#56A9F6 #73BE49 #F4D23E #F18E2E #EA5E5B #B26EDF #DDDEE0 #53585F'.split()


def hex_to_rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

def read_vote_data(votingfile):
    features = None
    states = []
    abbr = []
    matrix = []
    with open(votingfile, 'rU') as ins:
        for line in ins:
            bits = line.strip().split(',')
            if features is None:
                features = bits[2:]
            else:
                states.append(bits[0])
                abbr.append(bits[1])
                matrix.append([float(x) for x in bits[2:]])
    return features, states, abbr, np.array(matrix)

def make_map_file(mapfile, state_cluster_map, num_clusters=None):
    num_clusters = num_clusters or max(state_cluster_map.values())+1
    svg = open(mapfile, 'r').read()
    soup = BeautifulSoup(svg, "html5lib")
    paths = soup.findAll('path')
    for p in paths:
        if p['id'] in state_cluster_map.keys():
            dist = list(state_cluster_map[p['id']])
            dist = [math.sqrt(math.sqrt(math.sqrt(math.sqrt(x)))) for x in dist]
            dist = [x / sum(dist) for x in dist]
            (r, g, b) = (0., 0., 0.)
            for i, prob in enumerate(dist):
                (r_, g_, b_) = hex_to_rgb(COLORS[i])
                r += prob * r_
                g += prob * g_
                b += prob * b_
            color = str(rgb_to_hex(r, g, b))
            p['style'] = 'fill:%s;display:inline' % color
    f = open('figs/gmm_%d.svg' % num_clusters,"w")
    f.write(soup.prettify())
    f.close()


parser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
parser.add_argument('-n', action='store', dest='num_clusters', type=int, default=4, help='number of clusters')

if __name__ == '__main__':

    args = parser.parse_args()

    features, states, abbr, X = read_vote_data('data/state_vote_data.csv')

    # cluster the data
    gmm = GaussianMixture(n_components=args.num_clusters, covariance_type='spherical', max_iter=5, init_params='random', random_state=0).fit(X)

    # print cluster assignment distributions for each state
    preds = gmm.predict_proba(X)
    entropy = 0.
    for i, st in enumerate(states):
        print '%s\t%s\t%s' % (abbr[i], '{:<30}'.format(st), str(preds[i]))
        for x in preds[i]:
            try:
                entropy -= x * math.log(x, 2)
            except:
                pass
    entropy /= len(states)
    print 'entropy:', entropy

    # print mean values for each cluster
    for k, c in enumerate(gmm.means_):
        vector = dict(zip(features, c))
        print '\nCLUSTER %d' % k
        print '\t'.join(['']+[str(x) for x in range(1980,2017,4)])
        for party in 'dem rep 3rd'.split():
            dat = ['%.2f' % vector['%d_%s' % (year, party)] for year in range(1980,2017,4)]
            print '\t'.join([party]+dat)

    # visualize clusters in a map
    make_map_file('figs/Blank_US_Map_with_borders.svg', dict(zip(abbr, preds)), args.num_clusters)

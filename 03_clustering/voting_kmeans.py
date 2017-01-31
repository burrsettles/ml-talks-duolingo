"""
Burr Settles
Duolingo ML Dev Talk #3: Clustering

k-Means clustering example using scikit-learn.
"""

import argparse

import numpy as np

from bs4 import BeautifulSoup
from sklearn.cluster import KMeans

# cluster colors (for map visualizations, up to 8)
COLORS = '#56A9F6 #73BE49 #F4D23E #F18E2E #EA5E5B #B26EDF #DDDEE0 #53585F'.split()


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
            p['style'] = 'fill:%s;display:inline' % COLORS[state_cluster_map[p['id']]]
    f = open('figs/kmeans_%d.svg' % num_clusters,"w")
    # f.write(soup.prettify(formatter="xml"))
    f.write(soup.prettify())
    f.close()


parser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
parser.add_argument('-n', action='store', dest='num_clusters', type=int, default=5, help='number of clusters')

if __name__ == '__main__':

    args = parser.parse_args()

    features, states, abbr, X = read_vote_data('data/state_vote_data.csv')

    # cluster the data
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(X)

    # print cluster assignments for each state
    kmeans.labels_
    for i, st in enumerate(states):
        print '%s\t%s\t%s' % (kmeans.labels_[i], abbr[i], st)

    # print mean values for each cluster
    for k, c in enumerate(kmeans.cluster_centers_):
        vector = dict(zip(features, c))
        print '\nCLUSTER %d' % k
        print '\t'.join(['']+[str(x) for x in range(1980,2017,4)])
        for party in 'dem rep 3rd'.split():
            dat = ['%.2f' % vector['%d_%s' % (year, party)] for year in range(1980,2017,4)]
            print '\t'.join([party]+dat)

    # visualize clusters in a map
    make_map_file('figs/Blank_US_Map_with_borders.svg', dict(zip(abbr, kmeans.labels_)), args.num_clusters)

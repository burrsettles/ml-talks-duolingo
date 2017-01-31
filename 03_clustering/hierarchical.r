# Burr Settles
# Duolingo ML Dev Talk #3: Clustering

dat <- read.csv('data/state_vote_data.csv')
rownames(dat) <- dat$state
clustering <- hclust(dist(dat))

# dendrogram
plot(clustering, hang = -1)

# circular/fan plot
library(ape)
library(cluster)
plot(as.phylo(clustering), type = "fan")

# radial tree
plot(as.phylo(clustering), type = "unrooted")


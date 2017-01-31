"""
Burr Settles
Duolingo ML Dev Talk #1: Supervised Learning

Predicting online movie reviews with linear regression and scikit-learn.
"""

import math
import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.feature_extraction import DictVectorizer

def experiment(model_class, vectorizer, xval):
    name = model_class.__class__.__name__
    model = model_class.fit(X, y)
    model_weights = vectorizer.inverse_transform(model.coef_)[0]
    with open('weights.%s.txt' % name, 'w') as f:
        f.write('%s\t%f\n' % ('(intercept)', model.intercept_))
        f.writelines('%s\t%f\n' % k for k in model_weights.items())
    r2_scores = cross_validation.cross_val_score(model, X, y, scoring='r2', cv=xval)
    mae_scores = cross_validation.cross_val_score(model, X, y, scoring='mean_absolute_error', cv=xval)
    print '-'*80
    print 'r2\t%.4f\t%s' % (np.mean(r2_scores), name)
    print 'mae\t%.4f\t%s' % (np.mean(mae_scores), name)

if __name__ == "__main__":

    # read in the data set
    ratings = []
    reviews = []
    with open('data/review-dataset.txt', "rb") as data_file:
        for line in data_file:
            tokens = line.split()
            ratings.append(float(tokens[0]))
            reviews.append(dict((k, 1.) for k in set(tokens[1:])))

    # convert to scipy objects
    y = np.array(ratings)
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(reviews).toarray()

    # 3-fold cross validation with consistent seed
    kf = cross_validation.KFold(len(reviews), n_folds=3, shuffle=True, random_state=42)

    # OLS regression
    experiment(linear_model.LinearRegression(), vectorizer, kf)

    # L1/LASSO regression
    experiment(linear_model.Lasso(.001), vectorizer, kf)

    # L2/ridge regression
    experiment(linear_model.Ridge(100.), vectorizer, kf)
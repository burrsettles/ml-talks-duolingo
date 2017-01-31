"""
Burr Settles
Duolingo ML Dev Talk #1: Supervised Learning

Predicting ham vs. spam email with logistic regression and scikit-learn.
"""

import math
import numpy as np
from sklearn import cross_validation, linear_model
from sklearn.feature_extraction import DictVectorizer

def experiment(model_class, vectorizer, xval):
    name = model_class.__class__.__name__ + '.' + model_class.penalty
    model = model_class.fit(X, y)
    model_weights = vectorizer.inverse_transform(model.coef_)[0]
    with open('weights.%s.txt' % name, 'w') as f:
        f.write('%s\t%f\n' % ('(intercept)', model.intercept_))
        f.writelines('%s\t%f\n' % k for k in model_weights.items())
    acc_scores = cross_validation.cross_val_score(model, X, y, cv=xval)
    auc_scores = cross_validation.cross_val_score(model, X, y, scoring='roc_auc', cv=xval)
    prec_scores = cross_validation.cross_val_score(model, X, y, scoring='precision', cv=xval)
    recall_scores = cross_validation.cross_val_score(model, X, y, scoring='recall', cv=xval)
    f1_scores = cross_validation.cross_val_score(model, X, y, scoring='f1', cv=xval)
    print '-'*80
    print 'acc\t%.4f\t%s' % (np.mean(acc_scores), name)
    print 'auc\t%.4f\t%s' % (np.mean(auc_scores), name)
    print 'prec\t%.4f\t%s' % (np.mean(prec_scores), name)
    print 'recall\t%.4f\t%s' % (np.mean(recall_scores), name)
    print 'f1\t%.4f\t%s' % (np.mean(f1_scores), name)

if __name__ == "__main__":

    # read in the data set
    labels = []
    emails = []
    with open('data/SMSSpamCollection.txt', "rb") as data_file:
        for line in data_file:
            tokens = line.lower().split()
            labels.append(1 if tokens[0] == 'spam' else 0)
            emails.append(dict((k, 1.) for k in set(tokens[1:])))

    # convert to scipy objects
    y = np.array(labels)
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(emails).toarray()

    # 3-fold cross validation with consistent seed
    kf = cross_validation.KFold(len(emails), n_folds=3, shuffle=True, random_state=42)

    # L1-regularized regression
    experiment(linear_model.LogisticRegression(penalty='l1'), vectorizer, kf)

    # L2-regularized regression
    experiment(linear_model.LogisticRegression(penalty='l2'), vectorizer, kf)

from sklearn import svm
import numpy as np

from sklearn.externals import joblib

X = joblib.load("X.pkl")
Y = joblib.load("Y.pkl")
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
joblib.dump(clf, 'classifier.pkl')

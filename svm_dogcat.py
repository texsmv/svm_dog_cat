from sklearn import svm
import numpy as np


X = [[0,0],[1,1]]
y = [0,1]
clf = svm.SVC()
clf.fit(X,y)

print(clf.predict([[-1,-1]]))

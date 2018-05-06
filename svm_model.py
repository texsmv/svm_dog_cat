from sklearn import svm
import numpy as np
from sklearn.externals import joblib

T_c = joblib.load("Tc.pkl")
T_d = joblib.load("Td.pkl")
clf = joblib.load("classifier.pkl")

print("Prediciendo")

r_c = clf.predict([e for e in T_c])
r_c_t = len(r_c)
r_c_c = list(r_c).count(0)
r_d = clf.predict([e for e in T_d])
r_d_t = len(r_d)
r_d_c = list(r_d).count(1)
print("Total de imagenes de gatos", r_c_t)
print("Aciertos",r_c_c)
print("Porcentaje: ", r_c_c * 100 /r_c_t,"%")
#print(r_c)
print("Total de imagenes de perros", r_d_t)
print("Aciertos",r_d_c)
print("Porcentaje: ", r_d_c * 100 /r_d_t,"%")
#print(r_d)

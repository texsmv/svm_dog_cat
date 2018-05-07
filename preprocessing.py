from funciones import *


cats, dogs = script()
X = np.concatenate([cats, dogs])
print(np.shape(X))
Y = np.array([0] * len(cats) + [1] * len(dogs))
print(np.shape(Y))
T_c , T_d = script2()
joblib.dump(X, "X.pkl")
joblib.dump(Y, "Y.pkl")
joblib.dump(T_c, "Tc.pkl")
joblib.dump(T_d, "Td.pkl")

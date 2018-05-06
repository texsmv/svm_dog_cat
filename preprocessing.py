import numpy as np
import pywt
import cv2
import os
import imutils
from sklearn.externals import joblib


def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


def feature_vector(image):
    feature_vector = np.array([])
    R = []
    G = []
    B = []
    for i in image:
        for j in i:
            R.append(j[0])
            G.append(j[1])
            B.append(j[2])
    maxs = [max(R), max(G), max(B)]
    mins = [min(R), min(G), min(B)]
    feature_vector = np.concatenate([feature_vector, maxs])
    feature_vector = np.concatenate([feature_vector, mins])
    means, stds = cv2.meanStdDev(image)
    means_stds = np.concatenate([means, stds]).flatten()
    feature_vector = np.concatenate([feature_vector, means_stds])
    feature_vector = np.concatenate([feature_vector, extract_color_histogram(image)])
    return feature_vector


def feature_vector_4_img(img):
    return np.concatenate([feature_vector(e) for e in img])


def resize(mat):
    mat = cv2.resize(mat, (256, 256))
    return mat


def wavelet(mat):
    mat = np.float32(mat)
    m, (n, o, p) = pywt.dwt2(mat, "haar")
    m = np.uint8(m)
    (m, n, o, p) = (np.uint8(e) for e in (m, n, o, p))
    return m, (n, o, p)


def wvt(img):
    b, g, r = cv2.split(img)

    b, b_p = wavelet(b)
    g, g_p = wavelet(g)
    r, r_p = wavelet(r)
    img2 = cv2.merge([b, g, r])
    p = ([cv2.merge([b_p[i], g_p[i], r_p[i]]) for i in range(0, len(b_p))])

    return img2, p


def process_image(mat):
    mat = resize(mat)
    im, _ = wvt(mat)
    im, _ = wvt(im)
    img, p = wvt(im)
    return [img, p[0], p[1], p[2]]


def script():
    cats_path = os.listdir("cats/")
    dogs_path = os.listdir("dogs/")
    cats_imgs = [cv2.imread("cats/" + cats_path[i]) for i in range(0, len(cats_path))]
    dogs_imgs = [cv2.imread("dogs/" + dogs_path[i]) for i in range(0, len(dogs_path))]
    cats_imgs_haar = [process_image(e) for e in cats_imgs if e is not None]
    dogs_imgs_haar = [process_image(e) for e in dogs_imgs if e is not None]
    cats_vcs = [feature_vector_4_img(e) for e in cats_imgs_haar]
    dogs_vcs = [feature_vector_4_img(e) for e in dogs_imgs_haar]
    return cats_vcs, dogs_vcs


def script2():
    cats_path = os.listdir("test/cats/")
    dogs_path = os.listdir("test/dogs/")
    cats_imgs = [cv2.imread("test/cats/" + cats_path[i]) for i in range(0, len(cats_path))]
    dogs_imgs = [cv2.imread("test/dogs/" + dogs_path[i]) for i in range(0, len(dogs_path))]
    cats_imgs_haar = [process_image(e) for e in cats_imgs if e is not None]
    dogs_imgs_haar = [process_image(e) for e in dogs_imgs if e is not None]
    cats_vcs = [feature_vector_4_img(e) for e in cats_imgs_haar]
    dogs_vcs = [feature_vector_4_img(e) for e in dogs_imgs_haar]
    return cats_vcs, dogs_vcs


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

import cv2
import numpy as np

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
    maxs = [max(R),max(G),max(B)]
    mins = [min(R),min(G),min(B)]
    feature_vector = np.concatenate([feature_vector, maxs])
    feature_vector = np.concatenate([feature_vector, mins])
    means, stds = cv2.meanStdDev(image)
    means_stds = np.concatenate([means, stds]).flatten()
    feature_vector = np.concatenate([feature_vector, means_stds])
    return feature_vector


image = cv2.imread("cats/gato.jpg")
image2 = cv2.imread("dogs/perro.jpg")
print(image.shape)
print(feature_vector(image))
print(feature_vector(image2))

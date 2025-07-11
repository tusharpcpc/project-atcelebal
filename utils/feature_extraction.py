import cv2
import numpy as np
from skimage.feature import hog

def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    # HSV Histogram
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    color_hist = np.concatenate([h_hist, s_hist, v_hist])

    # HOG
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2),
                   orientations=9, block_norm='L2-Hys', feature_vector=True)

    # Edge Density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / 255.0

    return np.concatenate([color_hist, hog_feat, [edge_density]])

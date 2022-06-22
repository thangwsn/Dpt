from msilib.schema import Feature
import os
from hog import hog
import cv2

from lbp import lbp

IMG = 'person.jpg'
def main(img_path):
    l = lbp(IMG)
    h = hog(IMG).tolist()
    feature_all = [y for x in [l, h] for y in x]
    print('Extracted feature vector of %s. Shape:' % img_path)
    print("feature all size: ", len(feature_all))
    print(str(feature_all))
    pass

if __name__ == "__main__":
    main(IMG)
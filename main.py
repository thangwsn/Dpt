import os
from hog import hog
import cv2

from lbp import lbp

IMG = 'person.jpg'
def main(img_path):
    f = hog(IMG)
    g = lbp(IMG)

    print('Extracted feature vector of %s. Shape:' % img_path)
    print('Features (HOG):', len(f.tolist()))
    print("Feature (LBP): ", len(g))
    pass

if __name__ == "__main__":
    main(IMG)
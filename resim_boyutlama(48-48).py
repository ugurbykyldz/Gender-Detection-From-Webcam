import os
import cv2
from glob import glob

imgs = glob("train\man\*") #woman
for i in imgs:
    print(i)
    img = cv2.imread(i)
    img = cv2.resize(img,(48,48))
    os.remove(i)
    cv2.imwrite(i, img)

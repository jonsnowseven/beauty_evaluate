import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.models import load_model
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import time
from keras.preprocessing import image                  
from tqdm import tqdm
from model import *
import math


#################################################################################################


color = [(255,0,0),(0,255,255),(0,255,0),(0,100,255),(0,0,255)][::-1]

""" Main Functions """
def main():
	array = sys.argv[1:]
	if not len(array): 
		print('You need to put an image as argument input')
		sys.exit(0)
	if not os.path.exists(array[0]): 
		print('The path given does not exists')
		sys.exit(0)
	imagepath = array[0]

	face_clf = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	Model = Beau()
	Model.model.load_weights('beau.h5')
	image_name = imagepath.split('/')[-1]

	img = cv2.imread(imagepath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_clf.detectMultiScale(gray, 1.3, 5)
	
	for (x, y, w, h) in faces:
		fc = gray[y:y+h, x:x+w]

		roi = cv2.resize(fc, (160, 160))
		roi = roi.reshape((1,160,160,1))
		pred = Model.model.predict(roi)[0][0]
	
		cv2.putText(img, str(pred), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color[math.floor(pred)], 2)
		cv2.rectangle(img,(x,y),(x+w,y+h),color[math.floor(pred)],2)

	print(pred)
	cv2.imwrite('out'+image_name,img)



#################################################################################################
if __name__ == "__main__":
	main()


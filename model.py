#Imported libs
import pickle
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.optimizers import Adam
from tqdm import tqdm
import cv2
import pandas as pd
import os
import random

#from hyperas import optim
#from hyperas.distributions import choice, uniform
#from hyperopt import Trials, STATUS_OK, tpe


##################################################################################################################

def get_labels(file):
	with open(file,'r') as f:
		lines = f.readlines()
		return np.array([line.strip().split() for line in lines])
	
def index_process(flo):
	below = math.floor(flo)
	top = math.ceil(flo)
	
	result = np.zeros(5)
	result[below-1] = top-round(flo)
	result[top-1] = round(flo)-below
	
	return result


def load_data():
	#loading of images and labels
	#Cross validation was made by the dataset provider
	PATH = 'Beau/train_test_files/split60/'
	train_labels = get_labels(PATH+'train.txt')
	test_labels = get_labels(PATH+'test.txt')

	PATH2 = 'Beau/Images/'
	X_pre_train = [cv2.imread(PATH2+file,0) for file in tqdm(train_labels[:,0])]
	X_train = np.array([cv2.resize(img,(160,160),interpolation=cv2.INTER_AREA) for img in X_pre_train])

	X_pre_test = [cv2.imread(PATH2+file,0) for file in tqdm(test_labels[:,0])]
	X_test = np.array([cv2.resize(img,(160,160),interpolation=cv2.INTER_AREA) for img in X_pre_test])

	y_train = np.array([float(flo) for flo in train_labels[:,1]])
	y_test = np.array([float(flo) for flo in test_labels[:,1]])

	X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
	X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

	return X_train,X_test,y_train,y_test

##################################################################################################################
class Beau:
	def __init__(self):

		self.model = Sequential()

		#1st 2dConvolutional Layer
		self.model.add(Conv2D(64, (3, 3), padding='same', input_shape=(160, 160, 1)))
		self.model.add(Activation('relu'))

		#1st 2dMaxPool Layer
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))

		#2nd 2dConvolutional Layer
		self.model.add(Conv2D(64, (3, 3), padding='same'))
		self.model.add(Activation('relu'))

		#2nd 2dMaxPool Layer
		self.model.add(MaxPooling2D(pool_size=(2, 2)))  
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))

		#3rd 2dConvolutional Layer
		self.model.add(Conv2D(128, (5, 5), padding='same'))
		self.model.add(Activation('relu'))

		#3rd 2dMaxPool Layer
		self.model.add(MaxPooling2D(pool_size=(2, 2)))  
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.5))
		
		#4th 2dConvolutional Layer
		self.model.add(Conv2D(128, (5, 5), padding='same'))
		self.model.add(Activation('relu'))

		#3rd 2dMaxPool Layer
		self.model.add(MaxPooling2D(pool_size=(2, 2)))  
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))
		
		#5th 2dConvolutional Layer
		self.model.add(Conv2D(256, (3, 3), padding='same'))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())

		#3rd 2dMaxPool Layer
		self.model.add(MaxPooling2D(pool_size=(2, 2)))  
		self.model.add(BatchNormalization())
		self.model.add(Dropout(0.25))
		
		#6th 2dConvolutional Layer
		self.model.add(Conv2D(512, (5, 5), padding='same'))
		self.model.add(Activation('relu'))
		self.model.add(BatchNormalization())

		self.model.add(Flatten())

		#1st FC Layer
		self.model.add(Dense(256))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.25))
		
		#2nd FC Layer
		self.model.add(Dense(512))
		self.model.add(BatchNormalization())
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.25))

		# Output layer
		self.model.add(Dense(1))
		self.model.add(Activation('relu'))

		self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
		self.model.summary()

if __name__ == '__main__':
	
	X_train,X_test,y_train,y_test = load_data()
	model = Beau().model
	history = model.fit(X_train, y_train,
				batch_size=128,
				epochs=40,
				verbose=1,
				validation_data=(X_test, y_test))

	model.save('beau.h5')
	print(history.history.keys())
	with open('history.pickle','wb') as handle:
		pickle.dump(history.history, handle)
	

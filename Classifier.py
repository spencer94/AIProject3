import sys
import pandas as pd
import numpy as np
import pylab as pl
import os
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

STANDARD_SIZE = (100, 100)
DATA_DIR = "Images/"
TEST_FILE = "Images/TestImages/Hashtag.jpg"

def imgToArray(filename):
	
	try:
		img = Image.open(filename)
		img = img.resize(STANDARD_SIZE)
		img = img.convert('L')
		img = np.array(img)
		img = img.ravel()
	except IOError:
		print("Input File Error.")
		print("Verify file is .jpg and Check Path: "+inputFile)
		print("Exiting...")
		exit()
		
	return img

def main():

	if (len(sys.argv) != 2):
		print("Invalid number of Arguments. Check Syntax.")
		print("Syntax: Classifier.py <input img>")
		print("Where <input img> = File path to single image")
		print("Exiting...")
		exit()
	else:
		print("Processing Input Image...")
		TEST_FILE = str(sys.argv[1])

	print ("TRAINING STARTED!")

	print ("pulling images from files...")
	images = []
	labels = []

	#put training images in array
	for dirpath, dirnames, filenames in os.walk(DATA_DIR):
		for file in filenames:
			if (dirpath.split('/')[1]) != "TestImages":
				labels.append(dirpath.split('/')[1])
				images.append(imgToArray(os.path.join(dirpath, file)))
				
	x = np.array(images)
	y = np.array(labels)
	
	img = imgToArray(TEST_FILE)
	ImageToClassify = img.reshape(1,-1)
	
	print ("reducing arrays using randomizedPCA...")
	pca = RandomizedPCA(n_components=4)
	images = pca.fit_transform(images)
	ImageToClassify = pca.transform(ImageToClassify)

	print ("using K-closest neighbors to classify data...")
	knn = KNeighborsClassifier()
	knn.fit(images, labels)

	print ("-----------------------------------")
	print ("TESTING STARTED!")

	preds = knn.predict(ImageToClassify)
	print("Input Image Classfied as: " + preds[0])


main()
import os
import cv2
import pandas as pd
import numpy as np

def load_datasets(image_size,csv_path,train_path):
	images=os.listdir(train_path)
	X=[]
	image_size=128
	for img in images:
		img=os.path.join(train_path,img)
		image = cv2.imread(img)
		image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
		image = image.astype(np.float32)
		image = np.multiply(image, 1.0 / 255.0)
		X.append(image)
	X_train=np.array(X)
	#X_train.shape
	Y=pd.read_csv(csv_path)
	#Y.columns
	Y_train=Y["breed"]
	Y_train=np.array(Y_train).reshape(10222,1)
	return (X_train,Y_train)



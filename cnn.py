from google.cloud import storage

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to label binarizer")
#ap.add_argument("-i", "--image", required=True,
#	help="path to input image")
#args = vars(ap.parse_args())

## load the image
#image = cv2.imread(args["image"])
#output = imutils.resize(image, width=400)
# 
## pre-process the image for classification
#image = cv2.resize(image, (96, 96))
#image = image.astype("float") / 255.0
#image = img_to_array(image)
#image = np.expand_dims(image, axis=0)
#
## load the trained convolutional neural network and the multi-label
## binarizer
#print("[INFO] loading network...")
#model = load_model(args["model"])
#mlb = pickle.loads(open(args["labelbin"], "rb").read())
#
## classify the input image then find the indexes of the two class
## labels with the *largest* probability
#print("[INFO] classifying image...")
#proba = model.predict(image)[0]
#idxs = np.argsort(proba)[::-1][:2]
#
## loop over the indexes of the high confidence class labels
#for (i, j) in enumerate(idxs):
#	# build the label and draw the label on the image
#	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
#	cv2.putText(output, label, (10, (i * 30) + 25), 
#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
## show the probabilities for each of the individual labels
#for (label, p) in zip(mlb.classes_, proba):
#	print("{}: {:.2f}%".format(label, p * 100))
#
## show the output image
#cv2.imshow("Output", output)
#cv2.waitKey(0)

MODEL_BUCKET = os.environ['model_and_images_bucket']
MODEL_FILENAME = os.environ['fashion.model']
MODEL = None
def predict(image_path):
    #img_dir="C:\\"
    #image_path=img_dir+str(image_id)
	
    if not os.path.isfile("static/"+image_path):
	return "File dosen't exist"

    image = get_image(image_path)
    
	
    output = imutils.resize(image, width=400)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    print("[INFO] loading network...")
    model = g_load_model("fashion.model")
    mlb = pickle.loads(open( "mlb.pickle", "rb").read())
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
    	 label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    	 cv2.putText(output, label, (10, (i * 30) + 25), 
    		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	
    aperture=[(0,2.8),(2.8,4.9),(4.9),(0,215),(215,464),(464)]
    return "The recommended ISO settings are" +str(aperture[idxs[0]])+"recommended aperture settings are"+str(aperture[idxs[1]])
  
 
	
def get_image(image_id):
     
     client = storage.Client()
     bucket = client.get_bucket(MODEL_BUCKET)

     blob = bucket.blob(image_id)
    #bucket = storage.Client().get_bucket(bucket_name)

     return np.array(cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0))

image_path="00000004.jpg"
image=get_image(image_path)
print (image)



def g_load_model():
    #MODEL_BUCKET = os.environ['model_and_images_bucket']
    global MODEL
    client = storage.Client()
    bucket = client.get_bucket(MODEL_BUCKET)
    blob = bucket.get_blob(MODEL_FILENAME)
    s = blob.download_as_string()
    MODEL = pickle.loads(s)
    return MODEL
    

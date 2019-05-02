
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import random


def predict(image_path):
    #img_dir="C:\\"
    #image_path=img_dir+str(image_id)

    image_path = "static/"+image_path
    #image = cv2.imread(image_path)
    
    # output = imutils.resize(image, width=400)
    # image = cv2.resize(image, (96, 96))
    # image = image.astype("float") / 255.0
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)

    aperture=[(0,2.8),(2.8,4.9),(4.9)]
    iso=[(0,215),(215,464),(464)]
    a1=random.randint(0,2)
    i1=random.randint(0,2)
    print(a1)
    print(i1)
    ans="The recommended ISO settings are "+str(iso[i1])+"recommended aperture settings  are  "+str(aperture[a1])
    print(ans)
    return ans


    # print("[INFO] loading network...")
    # model = g_load_model()
   # mlb = pickle.loads(open( "mlb.pickle", "rb").read())
   #  proba = model.predict(image)[0]
    #    #  print(proba)
    #    #  idxs = np.argsort(proba)[::-1][:2]
    #    #  print(idxs)
    # for (i, j) in enumerate(idxs):
    #     # build the label and draw the label on the image
    # 	 label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    # 	 cv2.putText(output, label, (10, (i * 30) + 25),
    # 		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	#

    # show the probabilities for each of the individual labels
  
    
    # show the output image
    # cv2.imshow("Output", output)
    # cv2.waitKey(0)
# def get_image(image_id):
#
#      client = storage.Client()
#      bucket = client.get_bucket(MODEL_BUCKET)
#
#      blob = bucket.blob(image_id)
#     #bucket = storage.Client().get_bucket(bucket_name)
#
#      return np.array(cv2.imdecode(np.asarray(bytearray(blob.download_as_string())), 0))
#
#



# def g_load_model():
#     #MODEL_BUCKET = os.environ['model_and_images_bucket']
#     global MODEL
#     client = storage.Client()
#     bucket = client.get_bucket(MODEL_BUCKET)
#     blob = bucket.get_blob(MODEL_FILENAME)
#     print(type(blob))
#     s = blob.download_as_string()
#     MODEL = pickle.loads(s)
#     return MODEL
    




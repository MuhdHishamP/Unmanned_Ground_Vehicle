import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter


# Path to label map file
PATH_TO_LABELS = os.path.join('labelmap1.txt')

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# load model
interpreter = tf.lite.Interpreter(model_path="detect1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
while True:
# capture image
    ret, img_org = cap.read()
           
    modelpath='detect.tflite'  
    lblpath='labelmap.txt'
    min_conf=0.9
            # Grab filenames of all images in test folder
            #images = glob.glob(imgpath + '/.jpg') + glob.glob(imgpath + '/.JPG') + glob.glob(imgpath + '/.png') + glob.glob(imgpath + '/.bmp')
             
            #Load the label map into memory
    with open(lblpath, 'r') as f:
       labels1 = [line1.strip() for line1 in f.readlines()]
           
            #labels=["fire"]
            # Load the Tensorflow Lite model into memory
    interpreter1 = Interpreter(model_path=modelpath)
    interpreter1.allocate_tensors()
             
            # Get model details
    input_details1 = interpreter1.get_input_details()
    output_details1 = interpreter1.get_output_details()
    height = input_details1[0]['shape'][1]
    width = input_details1[0]['shape'][2]
             
    float_input1 = (input_details1[0]['dtype'] == np.float32)
             
    input_mean1 = 127.5
    input_std1 = 127.5
             
            # Randomly select test images
            #images_to_test = random.sample(images, num_test_images)
         

            # Load image and resize to expected shape [1xHxWx3]
            #image = cv2.imread(image_path)
    image1=img_org
    image_rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    imH1, imW1, _ = image1.shape
    image_resized1 = cv2.resize(image_rgb1, (width, height))
    input_data1 = np.expand_dims(image_resized1, axis=0)
             
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input1:
       input_data1 = (np.float32(input_data1) - input_mean1) / input_std1
             
            #print(input_details[0])    
            # Perform the actual detection by running the model with the image as input
    interpreter1.set_tensor(input_details1[0]['index'],input_data1)
    interpreter1.invoke()
             
            # Retrieve detection results
    boxes1 = interpreter1.get_tensor(output_details1[1]['index'])[0] # Bounding box coordinates of detected objects
    classes1 = interpreter1.get_tensor(output_details1[3]['index'])[0] # Class index of detected objects
    scores1 = interpreter1.get_tensor(output_details1[0]['index'])[0] # Confidence of detected objects
             
    detections = []
             
            # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores1)):
        if ((scores1[i] > min_conf) and (scores1[i] <= 1.0)):
             
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
           ymin1 = int(max(1,(boxes1[i][0] * imH1)))
           xmin1 = int(max(1,(boxes1[i][1] * imW1)))
           ymax1 = int(min(imH1,(boxes1[i][2] * imH1)))
           xmax1 = int(min(imW1,(boxes1[i][3] * imW1)))
                   
           cv2.rectangle(image1, (xmin1,ymin1), (xmax1,ymax1), (10, 255, 0), 2)
             
                   
                    # Draw label
           object_name1 = labels1[int(classes1[i])] # Look up object name from "labels" array using class index
           label1 = '%s: %d%%' % (object_name1, int(scores1[i]*100)) # Example: 'person: 72%'
           labelSize1, baseLine1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
           label_ymin1 = max(ymin1, labelSize1[1] + 10) # Make sure not to draw label too close to top of window
           cv2.rectangle(image1, (xmin1, label_ymin1-labelSize1[1]-10), (xmin1+labelSize1[0], label_ymin1+baseLine1-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
           cv2.putText(image1, label1, (xmin1, label_ymin1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
             
           detections.append([object_name1, scores1[i], xmin1, ymin1, xmax1, ymax1])

                   
               
    image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
    #cv2.imshow('image2', image1)

    # cv2.imshow('image', img_org)
    key = cv2.waitKey(1)
    if key == 27: # ESC
        break

    # prepara input image
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
    img = img.astype(np.uint8)

    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # run
    interpreter.invoke()

    # get outpu tensor
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    for i in range(boxes.shape[1]):
        if scores[0, i] > 0.5:
            box = boxes[0, i, :]
            x0 = int(box[1] * img_org.shape[1])
            y0 = int(box[0] * img_org.shape[0])
            x1 = int(box[3] * img_org.shape[1])
            y1 = int(box[2] * img_org.shape[0])
            box = box.astype(np.int)
            #
            #cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
            object_name = labels[int(classes[0, i])]
            #object_name = labels[int(classes[int(labels[0, i])])]
            if object_name=="knife" :
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                cv2.putText(img_org,object_name,(x0, y0),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
                print(object_name)

    # cv2.imwrite('output.jpg', img_org)
    cv2.imshow('image', img_org)

cap.release()
cv2.destroyAllWindows()

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import os
import cv2
import json
import argparse
import numpy as np
import tensorflow
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image Preprocessing, Reading Labels, and Loading model
with open('./models/labels.json', 'r') as labelFile:
    labels = labelFile.read()

labels = json.loads(labels)
# model = load_model('./models/googlenet.h5')
interpreter = tensorflow.lite.Interpreter(model_path='./models/mobilenet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def index(request):
    context = {'a': 1}
    return render(request, 'classifier/index.html', context)

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)

    test_image = '.' + filePathName
    # floating_model = input_details[0]['dtype'] == np.float32

    img = cv2.imread(test_image)
    img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
    img = np.expand_dims((img), axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    print(np.argmax(results))
    print(labels, end='\n')

    context = {'filePathName': filePathName, 'predictedLabel': labels[str(np.argmax(results))]}
    return render(request, 'classifier/predict.html', context)

def viewDatabase(request):
    images = os.listdir('./media/')
    imagesPath = ['./media/' + i for i in images]
    context = {'imagesPath': imagesPath}
    return render(request, 'classifier/viewDatabase.html', context)
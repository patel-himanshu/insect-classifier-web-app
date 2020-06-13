from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import Image
import json

# Image Preprocessing, Reading Labels, and Loading model
# img_height, img_width = 224, 224
with open('./models/labels.json', 'r') as labelFile:
    labels = labelFile.read()

labels = json.loads(labels)

model = load_model('./models/googlenet.h5')

def index(request):
    context = {'a': 1}
    return render(request, 'classifier/index.html', context)

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)

    img = Image.load_img(test, target_size=(224, 224))
    x = image
    context = {'filePathName': filePathName}
    return render(request, 'classifier/index.html', context)
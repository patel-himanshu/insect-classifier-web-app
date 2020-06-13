from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

import json
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

    test_image = '.' + filePathName
    # img = load_img(test_image, target_size=(224, 224))
    # img = img_to_array(img)
    # img = img/255
    # img = img.reshape(1, 224, 224, 3)

    # pred = model.predict(img)
    # predictedLabel = labels[str(np.argmax(pred[0]))]
    # print(pred)
    # print(np.argmax(pred[0]))
    # print(labels[str(np.argmax(pred[0]))])

    import cv2
    import numpy as np
    import argparse

    interpreter = tensorflow.lite.Interpreter(model_path='./models/mobilenet.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    floating_model = input_details[0]['dtype'] == np.float32

    img = cv2.imread(test_image)
    img = cv2.resize(img, (224, 224), cv2.INTER_AREA)
    img = np.expand_dims((img), axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    print(np.argmax(results))

    # top_k = results.argsort()[-5][::-1]
    # for i in top_k:
    #     if floating_model:
    #         print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
    #     else:
    #         print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
    
    print(labels, end='\n')
    # print(labels[i], end='\n')
    context = {'filePathName': filePathName, 'predictedLabel': labels[str(np.argmax(results))]}
    # context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
    return render(request, 'classifier/index.html', context)
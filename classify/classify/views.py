from django.http import response
from django.shortcuts import render
from django.core.files.storage import default_storage
from tensorflow import keras
import tensorflow as tf
import numpy as np
import os
import pathlib
import random
import string

model_path = os.path.join(os.getcwd(),'my_model.h5')
model_path = pathlib.Path(model_path)
cat_dog = tf.keras.models.load_model(model_path)

img_height = 180
img_width = 180
class_names = ['CAT','DOG']

def index(request):
    if request.method == "POST":
        f=request.FILES['sentFile']
        response = {}
        letters = string.ascii_lowercase
        file_name = ''.join(random.choice(letters) for i in range(3))
        file_name  += '.jpg'
        image_path = os.path.join(os.getcwd(),'images',file_name)
        image_path = pathlib.Path(image_path)
        file_name_2 = default_storage.save(image_path, f)
        original = keras.preprocessing.image.load_img(image_path, target_size=(img_height,img_width))
        img_array = keras.preprocessing.image.img_to_array(original)
        img_array = tf.expand_dims(img_array, 0)
        predictions = cat_dog.predict(img_array)
        score = tf.nn.softmax(predictions)
        result = class_names[np.argmax(score)]
        response['score'] = str(100*score)
        response['name'] = str(result)
        return render(request,'homepage.html', response)
    else:
        return render(request,'homepage.html')

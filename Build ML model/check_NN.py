import os
import tensorflow as tf
import pathlib
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import img_to_array

img_height = 180
img_width = 180
data_dir = os.path.join(os.getcwd(),'train')
data_dir = pathlib.Path(data_dir)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_dir = os.path.join(os.getcwd(),'test2','11.jpg')
test_dir = pathlib.Path(test_dir)

new_model = tf.keras.models.load_model('my_model.h5')

# img = image.load_img(data_dir, target_size=(img_height, img_width, 3))
# img_tensor = image.img_to_array(img)
# img_tensor /=255
# img_tensor = np.expand_dims(img_tensor, axis=0)
# test_im = new_model.predict(img_tensor)
# classes = np.argmax(test_im)
# print(classes)

img = keras.preprocessing.image.load_img(test_dir,target_size=(img_height,img_width))
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = new_model.predict(img_array)
score = tf.nn.softmax(predictions)
class_names = train_ds.class_names

print(class_names)
print(score)
print(class_names[np.argmax(score)])

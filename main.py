import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from database import get_features, insert_features
import os
from tensorflow.keras.models import Model
import cv2


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


model = load_model('model.keras')

feature_extractor = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d').output)

image = 'dataset/train/pants/01.jpg'


imgarray = cv2.imread(image)/255.
imgarray = tf.image.resize(imgarray, (224, 224))
imgarray = tf.expand_dims(imgarray, axis=0)

features = feature_extractor.predict(imgarray)
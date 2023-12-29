import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.utils import set_random_seed
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input
from scipy.spatial.distance import mahalanobis

path=os.path.dirname(__file__)
path=path+''
os.chdir(path)

size=(800,800,3)
set_random_seed(ord('h'))
def pre_model():

    base=ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_shape=size,
    )   
    flatten = layers.Flatten()(base.output)
    dense1 = layers.Dense(1024, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(512, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(512)(dense2)
    model= Model(base.input, output)
    
    return model


def preprocessing(image_path):
    image = tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image=preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image
    
    
def distance(output1,output2):
    dist = tf.norm(output1 - output2, ord=1, axis=-1)
    return dist

def main():
    dis=[]
    model=pre_model()
    main_image_path="test/main.jpg"
    input1=preprocessing(main_image_path)
    output1=model(input1)
    for i in os.listdir("test/"):
        print(i)
        image_path = os.path.join("test", i)
        input2=preprocessing(image_path)
        output2=model(input2)
        img_distance = distance(output1, output2).numpy().item() 
        dis.append((image_path, img_distance))
    sorted_images = sorted(dis, key=lambda x: x[1])

    for a, b in sorted_images:
        print(f"{a}, Distance: {b}")

main()
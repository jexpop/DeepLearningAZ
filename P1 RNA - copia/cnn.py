# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:49:36 2025

@author: JEXPO
"""

# Redes Neuronales Convolucionales

# Instalar Thenao
# pip install --upgrade --no-deps git+https://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras


# Parte 1 - Construir el modelo de CNN

# Importar las librerías y paquetes
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Inicializar la CNN
classifier = Sequential()

# Paso 2 - Convolucion
classifier.add(Convolution2D(32, 3, 3, 
               input_shape = (64, 64, 3),
               activation = "relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Paso 3 - Flattering
classifier.add(Flatten())

# Paso 4  - Full Connection
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

# Compilar la CNN
classifier.compile(optimizer = "adam", 
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])


# Parte 2 - Ajustar la CNN a las imágenes para entrenar

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('C:/Users/JEXPO/Downloads/dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('C:/Users/JEXPO/Downloads/dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)







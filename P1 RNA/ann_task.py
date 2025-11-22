# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 22:35:28 2025

@author: JEXPO
"""

# Redes Neuronales Artificiales - Reto


# Parte 1 - Pre procesado de datos

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categoricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer(
[('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],  
remainder='passthrough'                        
)
X = np.array(ct.fit_transform(X), dtype=float)
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Escalado de variables 
# (es obligatorio en redes neuronales)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Parte 2 - Construir la RNA

# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
# units = media entre nodos de entrada y de salida
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu", input_dim = 11))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform", 
                     activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform", 
                     activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", 
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados según la observación del reto
new_data = sc_X.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]]))
r_pred = classifier.predict(new_data)
r_pred = (r_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, r_pred)
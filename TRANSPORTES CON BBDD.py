# Databricks notebook source
#CONEXIÓN A LA BASE DE DATOS

import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName( "pandas to spark").getOrCreate()
from sklearn.metrics import r2_score



#conexión a la bbdd
jdbcHostname = "alesontest1.database.windows.net"
jdbcDatabase = "Prueba2"
jdbcPort = 1433
jdbcUrl = "jdbc:sqlserver://{0}:{1};database={2}".format(jdbcHostname, jdbcPort, jdbcDatabase)
connectionProperties = {
"user" : "aleson",
"password" : "QWaszx12345.",
"driver" : "com.microsoft.sqlserver.jdbc.SQLServerDriver"
}

#Nombre de la tabla que vamos a analizar y/o modificar
table="Mobile_combustions2"

Spdf = spark.read.jdbc(url=jdbcUrl, table=table, properties=connectionProperties)

#conversión a un DataFrame
df = Spdf.toPandas()



# COMMAND ----------

1. REGRESIÓ LINEAL PARA ESTIMAR EL COSTE DEL VIAJE

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# definir las variables predictoras y la variable objetivo
X = df[['Distance', 'Quantity', 'Fuel Quantity', 'Vehicle Type']]
y = df['Cost']

X = pd.get_dummies(X, columns=['Vehicle Type'])

X['Distance']=X['Distance'].astype(int)
X['Quantity']=X['Quantity'].astype(int)
X['Fuel Quantity']=X['Fuel Quantity'].astype(int)



# # dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crear el modelo de regresión lineal
model = LinearRegression()

# entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

#print("El X_test: \n", X_test.loc[0], "\n")

# hacer predicciones sobre el conjunto de prueba
j=0
prediccion=[]
for i in y:
    y_pred = model.predict([X.iloc[j]])    
    prediccion.append(y_pred[0])
    print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
    j+=1
    #next(x)
    
    
y_pred2 = model.predict(X)
accuracy = r2_score(y, y_pred2)
print('La precisión del modelo es:', accuracy*100)
    

# COMMAND ----------

#crear un dataframe con los inputs y outputs que se han sacado

resultados=pd.DataFrame()
resultados['Distance']=df['Distance']
resultados['Quantity']=df['Quantity']
resultados['Fuel Quantity']=df['Fuel Quantity']
resultados['Vehicle Type']=df['Vehicle Type']
resultados['output-deseado']=df['Cost']




# COMMAND ----------

prediccion=[round(i,2) for i in prediccion]


resultados['output-coste']=prediccion

resultados.head(5)

# COMMAND ----------

Spdf_overall=spark.createDataFrame(resultados)


# COMMAND ----------

Spdf_overall.write.jdbc(url=jdbcUrl, table='resultados_regresion_lineal', mode = "overwrite",properties=connectionProperties)

# COMMAND ----------

LO MISMO QUE EN EL ANTERIOR PERO CON REDES NEURONALES

# COMMAND ----------

#pip install tensorflow

# COMMAND ----------

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#X = pd.get_dummies(X, columns=['Contractual Instrument Type', 'Facility'])
data = df[['Distance', 'Quantity', 'Fuel Quantity', 'Cost']]
data=data.astype(float)

X=data.iloc[:, :-1].values
y=data.iloc[:,-1].values

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)
#X_test = scaler.transform(X)



# create neural network model, lo que hacemos aquí es añadir las capas a la red neuronal (numero de neuronas y la función que se aplica dentro)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, activation='relu', input_dim=X_train.shape[1]),
    
    tf.keras.layers.Dense(50, activation='linear'),
     tf.keras.layers.Dense(30, activation='linear'),
     tf.keras.layers.Dense(10, activation='linear'),
    
    tf.keras.layers.Dense(1)
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])


# entrenar el modelo
history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)


# test the model
predictions = model.predict(X)


for i in range(len(y)):
    print( '\n Valor real: ', round(y[i]), ' -- Valor estimado: ', round(predictions[i][0],2))
    
    
# evaluar el modelo
test_loss, test_mae = model.evaluate(X_test, y_test)
print("MAE en el conjunto de prueba:", test_mae)


# COMMAND ----------

len(predictions)

# COMMAND ----------

#crear un dataframe con los inputs y outputs que se han sacado

resultados2=pd.DataFrame()
resultados2['Distance']=df['Distance']
resultados2['Quantity']=df['Quantity']
resultados2['Fuel Quantity']=df['Fuel Quantity']

resultados2['output-deseado']=df['Cost']



# COMMAND ----------

pre=[round(i,2) for i in predictions]

pre

# COMMAND ----------

resultados2['output-coste'] = [round(float(i),2) for i in pre] 

resultados2.head(5)

# COMMAND ----------

resultados2.drop(['output-costes'], inplace= True, axis=1)

# COMMAND ----------

Spdf_overall=spark.createDataFrame(resultados2)


# COMMAND ----------

Spdf_overall.write.jdbc(url=jdbcUrl, table='resultados_redes_neuronales', mode = "overwrite",properties=connectionProperties)

# COMMAND ----------

REGRESIÓN LINEAL PARA ESTIMAR PERO SOLO SABIENDO DISTANCIA Y CANTIDAD

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




# definir las variables predictoras y la variable objetivo
X = df[['Distance','Quantity']]
y = df['Cost']


X['Distance']=X['Distance'].astype(int)
X['Quantity']=X['Quantity'].astype(int)
y=y.astype(int)

y=y.astype(int)

# # dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# crear el modelo de regresión lineal
model = LinearRegression()

# entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

#print("El X_test: \n", X_test.loc[0], "\n")

# hacer predicciones sobre el conjunto de prueba
j=0
for i in y_test:
    y_pred = model.predict([X_test.iloc[j]])    
    print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
    j+=1
    #next(x)
    
    
y_pred2 = model.predict(X)
accuracy = r2_score(y, y_pred2)
print('\n La precisión del modelo es: ', accuracy*100)
print("\n")
    

# COMMAND ----------

EL SIGUIENTE MODELO ES PARA PREDECIR EL TIPO DE COCHE A UTILIZAR

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# definir las variables predictoras y la variable objetivo
X = df[['Distance','Quantity', 'Fuel Type']]
y = df['Vehicle Type']


X['Distance']=X['Distance'].astype(int)
X['Quantity']=X['Quantity'].astype(int)

X = pd.get_dummies(X, columns=['Fuel Type'])


# # dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Crear modelo de regresión logística
modelo = LogisticRegression()

# Entrenar modelo con conjunto de datos de entrenamiento
modelo.fit(X, y)

# Predecir valores de la variable de salida en conjunto de datos de prueba
y_pred = modelo.predict(X)

# Calcular precisión del modelo en conjunto de datos de prueba
precision = accuracy_score(y, y_pred)
print('Precisión del modelo: {:.2f}'.format(precision))

j=0
for i in y:
    print("\nel valor real es: ", i, " el valor estimado es: ", y_pred[j])
    j+=1


# # hacer predicciones sobre el conjunto de prueba
# j=0
# for i in y_test:
#     y_pred = model.predict([X_test.iloc[j]])    
#     print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
#     j+=1
#     #next(x)
    
    

    

# COMMAND ----------

#crear un dataframe con los inputs y outputs que se han sacado

resultados3=pd.DataFrame()
resultados3['Distance']=df['Distance']
resultados3['Quantity']=df['Quantity']
resultados3['Fuel Type']=df['Fuel Type']



resultados3['output-deseado']=df['Vehicle Type']

# COMMAND ----------

resultados3['output-vehicle_type'] = y_pred 

resultados3.head(5)

# COMMAND ----------

Spdf_overall=spark.createDataFrame(resultados3)


# COMMAND ----------

Spdf_overall.write.jdbc(url=jdbcUrl, table='mobile_combustions_logistic_regresion', mode = "overwrite",properties=connectionProperties)

# COMMAND ----------

resultados4=pd.DataFrame()

cantidad=        [1000,1000,1000,1100,1500,2000,2000,2000,1900,2500,3000,2900,5000,3500,2500,6000]
tipo_combustible=['1' ,'2' ,'2' ,'2' ,'1' ,'1' ,'2' ,'2' ,'1' ,'1' ,'1' ,'1' ,'2' ,'1' ,'1' ,'2']
distancia=       [2500,3000,2000,580 ,1200,750 ,600 ,800 ,1000,5000,650 ,1455,2850,9874,500 ,205]
coste=           [6   ,6   ,4   ,2   ,3   ,3   ,4   ,5   ,6   ,8   ,4   ,6   ,8   ,10   ,6   ,5]
 
#crear un dataframe con los inputs y outputs que se han sacado

resultados4=pd.DataFrame()
resultados4['Distance']=distancia
resultados4['Quantity']=cantidad
resultados4['Fuel Type']=tipo_combustible

resultados4['Cost']=coste



# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score




# definir las variables predictoras y la variable objetivo
X = resultados4[['Distance', 'Quantity', 'Fuel Type']]
y = resultados4['Cost']

X = pd.get_dummies(X, columns=['Fuel Type'])

X['Distance']=X['Distance'].astype(int)
X['Quantity']=X['Quantity'].astype(int)



# # dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# crear el modelo de regresión lineal
model = LinearRegression()

# entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

#print("El X_test: \n", X_test.loc[0], "\n")

# hacer predicciones sobre el conjunto de prueba
j=0
prediccion=[]
for i in y:
    y_pred = model.predict([X.iloc[j]])    
    prediccion.append(y_pred[0])
    print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
    j+=1
    #next(x)
    
    
y_pred2 = model.predict(X)
accuracy = r2_score(y, y_pred2)
print('La precisión del modelo es:', accuracy*100)

# COMMAND ----------

prediccion=[round(i,2) for i in prediccion]


resultados4['output-coste']=prediccion

resultados4.head(5)

# COMMAND ----------

Spdf_overall=spark.createDataFrame(resultados4)
Spdf_overall.write.jdbc(url=jdbcUrl, table='ejemplo_visualizacion', mode = "overwrite",properties=connectionProperties)

# COMMAND ----------



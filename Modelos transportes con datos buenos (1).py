# Databricks notebook source
#CONEXIÓN A LA BASE DE DATOS



# COMMAND ----------

from pyspark.sql.functions import *


# COMMAND ----------


spa = spark.table("mobile_combustions_csv")
df = spa.toPandas()

spa = spark.table("emision_vehicle_type_distance_csv")
df2=spa.toPandas()

df.head(1)

df2.head(1)


# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# COMMAND ----------

# MAGIC %md
# MAGIC MODLEO QUE PASANDO LA DISTANCIA, EL TIPO DE VEHÍCULO Y LA CANTIDAD DE COMBUSTIBLE NOS PREDICE EL COSTE DEL VIAJE
# MAGIC     todo esto con los datos de ricoh que tienen sentido entre ellos

# COMMAND ----------


# definir las variables predictoras y la variable objetivo
X = df[['Distance',  'Vehicle Type', 'Fuel Quantity']]
y = df['Cost']

for i in range(len(y)):
    y[i]=float(y[i].replace(",","."))

X = pd.get_dummies(X, columns=['Vehicle Type'])

for i in range(len(X['Distance'])):
    X['Distance'][i]=float(X['Distance'][i].replace(",","."))
    
#X['Quantity']=X['Quantity'].astype(int)
for i in range(len(X['Fuel Quantity'])):
    X['Fuel Quantity'][i]=float(X['Fuel Quantity'][i].replace(",","."))



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
for i in y_test:
    y_pred = model.predict([X_test.iloc[j]])    
    prediccion.append(y_pred[0])
    print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
    j+=1
    #next(x)
    
    
y_pred2 = model.predict(X_test)
accuracy = r2_score(y_test, y_pred2)
print('La precisión del modelo es:', accuracy*100)

# COMMAND ----------

# MAGIC %md
# MAGIC En el bloque siguiente lo que se hace es dividir los vehículos entre los que se utilizan para transportar materiales y personas

# COMMAND ----------

#CON ESTO ME GUARDO EN vehiculos LOS VEHICULOS QUE SE UTILIZAN PARA EL FIN REQUERIDO
option=input("Que tipo de viaje se va a realizar (raw material(R)/personnel transport(P))? ")
while (option!='P' and option!='R'):
    option=input("Que tipo de viaje se va a realizar (raw material(R)/personnel transport(P))? ")

    
vehiculos=[]
for i in range(len(df2['Description'])):
    if option=='R' and 'raw' in df2['Description'][i]:
        vehiculos.append(df2['Vehicle Type'][i])
    if option=='P' and 'personnel' in df2['Description'][i]:
         vehiculos.append(df2['Vehicle Type'][i])
    
    

# COMMAND ----------

# MAGIC %md
# MAGIC Me guardo en un diccionario para cada vehículo el factor de emisión que hay que utilizar para convertir la distancia recorrida en emisiones

# COMMAND ----------

emisiones={}


for i in range(len(df2['Description'])):
    emisiones[df2['Vehicle Type'][i]]=[float(df2['CH4'][i].replace(",", ".")), float(df2['N20'][i].replace(",", "."))]
    

# COMMAND ----------

pip install pulp

# COMMAND ----------


# definir las variables predictoras y la variable objetivo
X = df[['Distance',  'Cost', 'Fuel Quantity']]
y = df['Vehicle Type']

#X = pd.get_dummies(X, columns=['Vehicle Type'])

X['Distance']=[float(i.replace(",",".")) for i in X['Distance']]
X['Cost']=[float(i.replace(",",".")) for i in X['Cost']]
X['Fuel Quantity']=[float(i.replace(",",".")) for i in X['Fuel Quantity']]




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
for i in y_test:
    y_pred = model.predict([X_test.iloc[j]])    
    prediccion.append(y_pred[0])
    print( ' Valor real:  ', i, ' Valor estimado: ', round(y_pred[0],2))
    j+=1
    #next(x)
    
    
y_pred2 = model.predict(X_test)
accuracy = r2_score(y_test, y_pred2)
print('La precisión del modelo es:', accuracy*100)

# COMMAND ----------

# MAGIC %md
# MAGIC Este modelo usando clasificación que da el coche que minimiza las emisiones en CH4

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Dividir los datos en un conjunto de entrenamiento y un conjunto de prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3)

# Construir el modelo de clasificación
modelo = DecisionTreeClassifier()
modelo.fit(X_entrenamiento, y_entrenamiento)

# Hacer una predicción
predicciones = modelo.predict(X_prueba)



# Calcular la precisión del modelo
precision = accuracy_score(y_prueba, predicciones)

print(precision*100)

# COMMAND ----------

# MAGIC %md
# MAGIC El siguiente código, dada una distancia te dice que tipo de vehiculo será el que menos emite, el problema de esto es que siempre da el mismo

# COMMAND ----------



# diccionario que contiene los datos de emisiones de CO2 de varios tipos de vehículos
emisiones_vehiculos = emisiones

# función que devuelve el tipo de vehículo con la menor cantidad de emisiones de CO2
def tipo_vehiculo_optimo(distancia):
    emisiones_minimas = float('inf')
    tipo_vehiculo_optimo = None
    for tipo_vehiculo, emisiones in emisiones_vehiculos.items():
        emisiones_totales = distancia*emisiones_vehiculos[tipo_vehiculo][0]
        if emisiones_totales < emisiones_minimas:
            emisiones_minimas = emisiones_totales
            tipo_vehiculo_optimo = tipo_vehiculo
    return tipo_vehiculo_optimo



tipo_vehiculo = tipo_vehiculo_optimo(1) # distancia de 10 kilómetros
print("El tipo de vehículo óptimo para recorrer 10 kilómetros es: ", tipo_vehiculo)

# COMMAND ----------

# MAGIC %md
# MAGIC EJEMPLO EN EL QUE SE LE PASA UNA DISTANCIA Y SI ES PARA TRANSPORTAR MATERIALES O PERSONAS Y DA TODOS LOS POSIBLES VEHÍCULOS Y CUANTO EMITE CADA UNO

# COMMAND ----------


distancia=input("Dame la distancia que hay que recorrer: ")
distancia=float(distancia)
option=input("Dame el tipo de viaje se va a realizar (raw material(R)/personnel transport(P)) ")
while (option!='P' and option!='R'):
    option=input("Dame el tipo de viaje se va a realizar (raw material(R)/personnel transport(P)) ")
    
    
print("Para recorrer ", distancia, "km se emitirá: \n")
for i in range(len(df2['Description'])):
    if option=='R' and 'raw' in df2['Description'][i]:
        [float(df2['CH4'][i].replace(",", ".")), float(df2['N20'][i].replace(",", "."))]
        print( "El vehículo: ",df2['Vehicle Type'][i], "emitirá de CH4: ",round(distancia*float(df2['CH4'][i].replace(",", ".")),2), "g, emitirá de N20", round(distancia*float(df2['N20'][i].replace(",", ".")),2), "g" )
    if option=='P' and 'personnel' in df2['Description'][i]:
         print( "El vehículo: ",df2['Vehicle Type'][i],  "emitirá de CH4: ",round(distancia*float(df2['CH4'][i].replace(",", ".")),2), "g, emitirá de N20", round(distancia*float(df2['N20'][i].replace(",", ".")),2), "g" )


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC PASANDO LA DISTANCIA Y LA CANTIDAD DE FUEL, NOS DEVUELVE PARA CADA TIPO DE VEHÍCULO EL COSTE Y CUANTO EMITIRÁ EN ESE VIAJE

# COMMAND ----------

import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

#guardamos la transformación en emisiones
emisiones={}
for i in range(len(df2['Description'])):
    emisiones[df2['Vehicle Type'][i]]=[float(df2['CH4'][i].replace(",", ".")), float(df2['N20'][i].replace(",", "."))]


#transformar los datos de string a float
X = df[['Distance',  'Fuel Quantity', 'Vehicle Type', 'Cost']]
for i in range(len(X['Distance'])):
    X['Distance'][i]=float(X['Distance'][i].replace(",","."))   
for i in range(len(X['Fuel Quantity'])):
    X['Fuel Quantity'][i]=float(X['Fuel Quantity'][i].replace(",","."))
for i in range(len(X['Cost'])):
    X['Cost'][i]=float(X['Cost'][i].replace(",","."))
    
    
df_vector=[]
tipos_vehiculo = list(X['Vehicle Type'].unique())
for tipo in tipos_vehiculo:
  df_vector. append(X[X['Vehicle Type'] == tipo])


modelos_vector=[]
for tipo in df_vector:
  modelo=LinearRegression()
  modelos_vector.append(modelo.fit(tipo[['Distance', 'Fuel Quantity']], tipo['Cost']))

distancia=483
combustible=1303.03
matriz=[]
diccionario={}

for i in range(len(modelos_vector)):
  costo= modelos_vector[i].predict([[distancia, combustible]])[0]
  emision_ch4=distancia*emisiones[tipos_vehiculo[i]][0]
  emision_n2o=distancia*emisiones[tipos_vehiculo[i]][1]
  print(f"El costo del viaje en {tipos_vehiculo[i]} es: {costo}. Además, este viaje se emitirá: ",emision_ch4, "g de CH4 y se emitirá: ",emision_n2o, "g de N20")
  vector=[tipos_vehiculo[i], costo,emision_ch4,emision_n2o]
  matriz.append(vector)
  diccionario[tipos_vehiculo[i]]=[costo,emision_ch4,emision_n2o]
    



# COMMAND ----------

# MAGIC %md
# MAGIC  DE TODAS LAS OPCIONES ANTERIORES ME QUEDO CON LA QUE TENGA EL MEJOR EQUILIBRIO, usando ponderaciones

# COMMAND ----------

peso_coste=0.5
peso_emisiones=0.5

puntuaciones=[]

for vehiculo in tipos_vehiculo:
    puntuacion=(peso_coste * diccionario[vehiculo][0]) + (peso_emisiones * diccionario[vehiculo][1])
    puntuaciones.append([vehiculo,puntuacion])

m=puntuaciones[0][1]
vehiculo=puntuaciones[0][0]
for punt in puntuaciones:
    if punt[1]<m:
        m=punt[1]
        vehiculo=punt[0]
        
    

print(f"El vehículo con mejor equilibrio entre costo y emisiones es: {vehiculo}")


# COMMAND ----------

# MAGIC %md
# MAGIC EXPRESAR EL RESULTADO EN FORMA DE TABLA

# COMMAND ----------

import matplotlib.pyplot as plt

fig, ax =plt.subplots(1,1)


column_labels=["Vehicle Type", "Cost", "CH4", "N20"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=matriz,colLabels=column_labels,loc="center")

plt.show()

# COMMAND ----------



# COMMAND ----------



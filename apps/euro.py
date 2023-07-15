import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn import metrics
import plotly.express as px

def app():
    st.title('Euros')

    tiempo = st.date_input('Introduzca una fecha anterior al 2021')
    tiempo = str(tiempo)
    anio_presente = (tiempo[:4])

    df = pd.read_csv('PeruTipoDeCambio.csv')

    df['Fecha_Cambio'] = pd.to_datetime(df['Fecha_Cambio'])
    df = df.set_index("Fecha_Cambio")
    df = df[anio_presente:]

    st.title('Predicción del tipo de cambio Sol-Euro')
    st.subheader('Datos con respecto al tipo de cambio Sol - Euro')

    if int(anio_presente) < 2021:
        st.write(df['TC_Soles_Euros'].describe())

    #Visualizaciones 

  
        st.subheader('Variacion del tipo de cambio del Euro en el tiempo')
        fig = plt.figure(figsize = (12,6))
        plt.plot(df.TC_Soles_Dolares)
        st.pyplot(fig)



        # Splitting data into training and testing 

        # Create a new dataframe with only the 'Close column 
        data = df.filter(['TC_Soles_Euros'])
        # Convert the dataframe to a numpy array
        dataset = data.values
        # Get the number of rows to train the model on
        training_data_len = int(np.ceil( len(dataset) * .95 ))


        #data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        #data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        
        
        #from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler(feature_range = (0,1))

        #data_training_array = scaler.fit_transform(data_training)




        # Cargar mi modelo

        model = load_model('keras_model2.h5')

        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002 
        # Crear el conjunto de datos de prueba
        # Crear una nueva matriz que contenga valores escalados del índice 1543 al 2002
        test_data = scaled_data[training_data_len - 60: , :]
        # Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        # Convert the data to a numpy array
        # Convierte los datos en una matriz numpy
        x_test = np.array(x_test)

        # Reshape the data
        # Reforma los datos
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        # Get the models predicted price values 
        # Obtenga los valores de precios predichos de los modelos
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Get the root mean squared error (RMSE)
        # Obtener la raíz del error cuadrático medio (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        rmse





        # Grafico Final
        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        st.subheader('Comparacion de entrenamiento, validacion y prediccion')
        fig2=plt.figure(figsize=(16,6))
        plt.title('Model')
        plt.xlabel('Tiempo', fontsize=18)
        plt.ylabel('Precio euro (€)', fontsize=18)
        plt.plot(train['TC_Soles_Euros'])
        plt.plot(valid[['TC_Soles_Euros', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        plt.show()
        st.pyplot(fig2)

        
        #Grafico
        st.subheader('Precio predecido vs Precio Original')
        fig3=plt.figure(figsize=(12,6))
        plt.plot(y_test, 'b', label = 'Precio Original')
        plt.plot(predictions, 'r', label= 'Precio Predecido')
        plt.xlabel('Tiempo')
        plt.ylabel('Precio')
        plt.legend()
        st.pyplot(fig3)

        st.subheader('Mostrar los datos originales y los datos predecidos') 
        st.write(valid)
        
         # Evaluación del modelo
    
        st.title('Evaluación del Modelo LSTM')
        ## Métricas
        MAE=metrics.mean_absolute_error(y_test, predictions)
        MSE=metrics.mean_squared_error(y_test, predictions)
        RMSE=np.sqrt(metrics.mean_squared_error(y_test, predictions))

        metricas = {
            'metrica' : ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
            'valor': [MAE, MSE, RMSE]
        }
        metricas = pd.DataFrame(metricas)  
        ### Gráfica de las métricas
        st.subheader('Métricas de rendimiento') 
        fig = px.bar(        
            metricas,
            x = "metrica",
            y = "valor",
            title = "Métricas del Modelo LSTM",
            color="metrica"
        )
        st.plotly_chart(fig)
        
    else:
        st.write('Solo se puede hasta el 2021')

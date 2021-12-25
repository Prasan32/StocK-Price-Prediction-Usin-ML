from json import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import time
from PIL import Image

start= '2011-01-01'
end='2021-12-20'

rad=st.sidebar.radio('Navigation',['Home','About'])


if rad=='About':
    st.spinner()
    with st.spinner(text='Loading...'):
      time.sleep(5)
      st.title('Stock Market Prediction Using Machine Learning')
      st.header('Stock Price Prediction')
      st.write('Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. ')
      st.header('Understanding Long Short Term Memory Network')
      st.write('Here, we have used a Long Short Term Memory Network (LSTM) for building our model to predict the stock prices of companies.')
      st.write('LTSMs are a type of Recurrent Neural Network for learning long-term dependencies. It is commonly used for processing and predicting time-series data. ')
      image=Image.open('LSTM architecture.PNG')
      st.image(image, caption='LSTM architecture')
      st.write('From the image on the top, you can see LSTMs have a chain-like structure. General RNNs have a single neural network layer. LSTMs, on the other hand, have four interacting layers communicating extraordinarily.')
      st.write('LSTMs work in a three-step process.')
      st.write('i.     The first step in LSTM is to decide which information to be omitted from the cell in that particular time step. It is decided with the help of a sigmoid function. It looks at the previous state (ht-1) and the current input xt and computes the function.')
      st.write('ii.    There are two functions in the second layer. The first is the sigmoid function, and the second is the tanh function. The sigmoid function decides which values to let through (0 or 1). The tanh function gives the weightage to the values passed, deciding their level of importance from -1 to 1.')
      st.write('iii.   The third step is to decide what will be the final output. First, you need to run a sigmoid layer which determines what parts of the cell state make it to the output. Then, you must put the cell state through the tanh function to push the values between -1 and 1 and multiply it by the output of the sigmoid gate.')

if rad=='Home':
  st.spinner()
  with st.spinner(text='Loading...'):
      time.sleep(5)  
  st.title('Stock Market Prediction Using Machine Learning')

  # user_input=st.text_input('Enter Stock Ticker','AAPL')
  user_input=st.selectbox('Choose Stock Ticker',('AAPL','TSLA','AMD','NVDA','INTC','GOOG'))
  df=data.DataReader(user_input, 'yahoo', start, end)

  # Describing Data
  st.subheader('Data from 2011-2021')
  st.write(df.describe())

  # Visualisation
  st.subheader('Closing Price VS Time Chart')
  fig=plt.figure(figsize=(12,6))
  plt.plot(df.Close)
  st.pyplot(fig)

  st.subheader('Closing Price VS Time Chart with 100MA')
  ma100=df.Close.rolling(100).mean()
  fig=plt.figure(figsize=(12,6))
  plt.plot(ma100)
  plt.plot(df.Close)
  st.pyplot(fig)

  st.subheader('Closing Price VS Time Chart with 100MA & 200MA')
  ma100=df.Close.rolling(100).mean()
  ma200=df.Close.rolling(200).mean()
  fig=plt.figure(figsize=(12,6))
  plt.plot(ma100,'r')
  plt.plot(ma200,'g')
  plt.plot(df.Close,'b')
  st.pyplot(fig)


  # splitting data into training and testing

  data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
  data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

  from sklearn.preprocessing import MinMaxScaler
  scaler=MinMaxScaler(feature_range=(0,1))

  data_training_array=scaler.fit_transform(data_training)

  st.spinner()
  with st.spinner(text='Loading LSTM model...'):
      time.sleep(5)
  # Load my model
  model=load_model('keras_model.h5')

  # Testing Part
  past_100_days=data_training.tail(100)
  final_df=past_100_days.append(data_testing,ignore_index=True)
  input_data=scaler.fit_transform(final_df)


  x_test=[]
  y_test=[]


  for i in range(100,input_data.shape[0]):
     x_test.append(input_data[i-100:i])
     y_test.append(input_data[i,0])

  x_test,y_test=np.array(x_test),np.array(y_test)
  st.spinner()
  with st.spinner(text='Predicting Data...'):
      time.sleep(5)  
  y_predicted=model.predict(x_test)
  scaler=scaler.scale_

  scale_factor=1/scaler[0]
  y_predicted=y_predicted*scale_factor
  y_test=y_test*scale_factor


  # Final Graph
  st.subheader('Predictions VS Original')
  st.spinner()
  with st.spinner(text='Loading the result...'):
      time.sleep(5)
  fig2=plt.figure(figsize=(12,6))
  plt.plot(y_test,'b',label='Original Price')
  plt.plot(y_predicted,'r',label='Predicted Price')
  plt.xlabel('Time')
  plt.ylabel('Price')
  st.pyplot(fig2)
  st.balloons()



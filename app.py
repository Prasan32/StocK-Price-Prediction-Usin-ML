import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import time
from PIL import Image
from streamlit.elements import form
from streamlit.legacy_caching.caching import cache

st.set_page_config(page_title='Stock Price Prediction', page_icon="🤑")

start= '2011-01-01'
end='2021-12-20'

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: 1 rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


rad=st.sidebar.selectbox('Navigation',('Home','Stock Price Prediction','Contact',))

st.sidebar.write("Visit Developer Profile:[Prasanna KB](https://www.linkedin.com/in/prasanna-kumar-baniya-9a91a5179/)")

# new_title = '<p style="font-family:sans-serif; color:#8271D2; font-size: 15px;margin-top: 320px"><b>Visit Developer Profile:</b><br><a style="text-decoration:none;color:#8271D2" href="https://github.com/Prasan32">Prasanna KB</a></p>'
# st.sidebar.markdown(new_title, unsafe_allow_html=True)

if rad=='Home':
    # json=pd.read_json('89023-loading-circles.json')
    st.spinner()
    with st.spinner(text='Loading...'):
      time.sleep(5)
      new_title = '<h1 style="font-family:sans-serif; color:#8271D2; font-size: 42px;">Stock Price Prediction Using Machine Learning</h1>'
      st.markdown(new_title, unsafe_allow_html=True)
      heading1 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Introduction</p>'
      st.markdown(heading1, unsafe_allow_html=True)
      st.write('Stock Price Prediction using machine learning helps you discover the future value of company stock and other financial assets traded on an exchange. The entire idea of predicting stock prices is to gain significant profits. Predicting how the stock market will perform is a hard task to do. There are other factors involved in the prediction, such as physical and psychological factors, rational and irrational behavior, and so on. All these factors combine to make share prices dynamic and volatile. This makes it very difficult to predict stock prices with high accuracy. ')
      heading1 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">LSTM</p>'
      st.markdown(heading1, unsafe_allow_html=True)      
      st.write('Here, we have used a Long Short Term Memory Network (LSTM) for building our model to predict the stock prices of companies.')
      st.write('LTSMs are a type of Recurrent Neural Network for learning long-term dependencies. It is commonly used for processing and predicting time-series data. ')
      image=Image.open('LSTM architecture.PNG')
      st.image(image, caption='LSTM architecture')
      st.write('From the image on the top, you can see LSTMs have a chain-like structure. General RNNs have a single neural network layer. LSTMs, on the other hand, have four interacting layers communicating extraordinarily.')
      st.write('LSTMs work in a three-step process.')
      st.write('i.     The first step in LSTM is to decide which information to be omitted from the cell in that particular time step. It is decided with the help of a sigmoid function. It looks at the previous state (ht-1) and the current input xt and computes the function.')
      st.write('ii.    There are two functions in the second layer. The first is the sigmoid function, and the second is the tanh function. The sigmoid function decides which values to let through (0 or 1). The tanh function gives the weightage to the values passed, deciding their level of importance from -1 to 1.')
      st.write('iii.   The third step is to decide what will be the final output. First, you need to run a sigmoid layer which determines what parts of the cell state make it to the output. Then, you must put the cell state through the tanh function to push the values between -1 and 1 and multiply it by the output of the sigmoid gate.')

if rad=='Stock Price Prediction':
  st.spinner()
  with st.spinner(text='Loading...'):
      time.sleep(5)  
  new_title = '<h1 style="font-family:sans-serif; color:#8271D2; font-size: 42px;">Stock Price Prediction Using Machine Learning</h1>'
  st.markdown(new_title, unsafe_allow_html=True)

  # user_input=st.text_input('Enter Stock Ticker','AAPL')
  user_input=st.selectbox('Choose Stock Ticker',('','AAPL','TSLA','AMD','NVDA','INTC','GOOG'))
  if user_input=='':
    st.warning('Please select a stock ticker')
  else:
    df=data.DataReader(user_input, 'yahoo', start, end)



    # Describing Data
    heading1 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Data from 2011-2021</p>'
    st.markdown(heading1, unsafe_allow_html=True)
    st.write(df.describe())

    # Visualisation
    heading2 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Closing Price VS Time Chart</p>'
    st.markdown(heading2, unsafe_allow_html=True)
    fig=plt.figure(figsize=(12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    heading3 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Closing Price VS Time Chart with 100MA</p>'
    st.markdown(heading3, unsafe_allow_html=True)
    ma100=df.Close.rolling(100).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    heading4 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Closing Price VS Time Chart with 100MA & 200MA</p>'
    st.markdown(heading4, unsafe_allow_html=True)
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
        time.sleep(4)
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
    heading5 = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;">Predictions VS Original</p>'
    st.markdown(heading5, unsafe_allow_html=True)
    st.spinner()
    with st.spinner(text='Loading the result...'):
        time.sleep(5)
    fig2=plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label='Original Price')
    plt.plot(y_predicted,'r',label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    st.pyplot(fig2)
    # st.balloons()

if rad=='Contact':
    new_title = '<p style="font-family:sans-serif; color:#8271D2; font-size: 30px;"><b>We would love to hear from you!</b><br>Send us a message!</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    with st.form("my_form",clear_on_submit=True):
       name=st.text_input(label='Name:',max_chars=50,placeholder='Enter your name')    
       email=st.text_input(label='Email:',placeholder='Enter your email address')
       msg=st.text_area(label='Message:',height=100)
       slider_val = st.slider("Rate us (1 to 5)",0,5,value=3)
    #    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
       submitted = st.form_submit_button("Submit")
    if submitted:
        # st.write("Rating", slider_val,"name",name)
        d = {"Name":name, 
            "Email":email,
            "Message":msg,
            "Rating":slider_val,}

        if name=="" and email=="" and msg=="":
            # st.markdown('<h5>Please fill all the fields<h5>',unsafe_allow_html=True)
            st.error('Please fill all the fields')
        else:
           st.markdown('<h3 style="font-family:sans-serif; color:#8271D2;">Thank you for your feedback!</h3>', unsafe_allow_html=True)  
           df=pd.read_json('response.json')
           df = df.append(d, ignore_index = True)
           open('response.json', 'w').write(df.to_json())


         


from cProfile import label
from logging import PlaceHolder
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

st.set_page_config(page_title='Stock Price Prediction', page_icon="🤑",initial_sidebar_state="collapsed",menu_items={
 'Get Help': 'https://www.extremelycoolapp.com/help',
 'Report a bug': "https://www.extremelycoolapp.com/bug",
 'About': "This is an Stock Price Prediction Application for future investment. This application is made for educational purpose only."
 })

start= '2011-01-01'
end='2021-12-20'

# st.markdown(""" <style>
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# </style> """, unsafe_allow_html=True)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: 1 rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


rad=st.sidebar.selectbox('Navigation',('Home','Stock Price Prediction','Contact',))


if rad=='Home':
    #   i=1
    #   while i<2:
    #       st.sidebar.write("")
    #       i+=1
    #   image = Image.open('stock.jpg')
    #   st.sidebar.image(image, caption='')
      new_title = '<h1 style="font-family:serif;  font-size: 72px;text-align:center;">Stock Market Price Prediction</h1><br><p style="text-align:center;font-family:serif;font-size: 32px;">Welcome! to the future of investing</p>'
      st.markdown(new_title, unsafe_allow_html=True)
      heading1 = '<p style="font-family:sans-serif; color:black; font-size: 15px;margin-top:110px"><b style="font-weight:bold">Disclaimer</b>:<br>We are not a financial expert. This project is for educational purposes only to demonstrate the application of TensorFlow/Keras,LSTM, Streamlit and other visualisations.Please consult a professional financial consultant for investing. Invest at your own risk.</p>'
      st.markdown(heading1, unsafe_allow_html=True)
   
if rad=='Stock Price Prediction':
  new_title = '<h1 style="font-family:serif; font-size: 42px;">Stock Price Prediction Using Machine Learning</h1>'
  st.markdown(new_title, unsafe_allow_html=True)
  
  dict={'APPLE':'AAPL','GOOGLE':'GOOG','TESLA':'TSLA','AMD':'AMD',"NVIDIA":'NVDA','INTEL':'INTC','FACEBOOK':'FB2A.BE'}

  with st.form("my_form1"):
    uploaded_file = st.file_uploader("Upload a csv file")
    st.write('OR')
    user_input=st.selectbox('Choose Stock Ticker',('','APPLE','TESLA','AMD','NVIDIA','INTEL','GOOGLE','FACEBOOK'))
    col1,col2=st.columns(2)
    start=col1.selectbox('From',('2010','2011','2012','2013','2014','2015'))
    end=col2.selectbox('To',('2016','2017','2018','2019','2020'))

    go=st.form_submit_button("Show")
  
  if go:
    if user_input:
        df=data.DataReader(dict[user_input], 'yahoo', start, end)

        # Describing Data
        heading1 = f"""<p style="font-family:serif; font-size: 30px;">Data from {start}-{end}</p>"""
        st.markdown(heading1, unsafe_allow_html=True)
        st.write(df.describe())

        # Visualisation
        heading2 = '<p style="font-family:serif; font-size: 30px;">Closing Price VS Time Chart</p>'
        st.markdown(heading2, unsafe_allow_html=True)
        fig=plt.figure(figsize=(12,6))
        plt.plot(df.Close,'b',label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        heading3 = '<p style="font-family:serif; font-size: 30px;">Closing Price VS Time Chart with 100MA</p>'
        st.markdown(heading3, unsafe_allow_html=True)
        ma100=df.Close.rolling(100).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r',label='100 days MA')
        plt.plot(df.Close,'b',label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        heading4 = '<p style="font-family:serif; font-size: 30px;">Closing Price VS Time Chart with 100MA & 200MA</p>'
        st.markdown(heading4, unsafe_allow_html=True)
        ma100=df.Close.rolling(100).mean()
        ma200=df.Close.rolling(200).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r',label='100 days MA')
        plt.plot(ma200,'g',label='200 days MA')
        plt.plot(df.Close,'b',label='Closing Price')
        plt.xlabel('Time ')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)


        # splitting data into training and testing

        data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))

        data_training_array=scaler.fit_transform(data_training)

        st.spinner()
        with st.spinner(text='Loading LSTM model...'):
            time.sleep(1)
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
            time.sleep(2) 
        y_predicted=model.predict(x_test)
        scaler=scaler.scale_

        mse=np.square(np.subtract(y_test,y_predicted)).mean()
        rmse=np.sqrt(mse)
        # rmse=np.sqrt(np.mean(((y_predicted-y_test)**2)))

        mape=np.mean(np.abs((y_test-y_predicted)/y_test))*100

        accuracy=100-(rmse*100)

        scale_factor=1/scaler[0]
        y_predicted=y_predicted*scale_factor
        y_test=y_test*scale_factor

            

        i=1
        while i<5:
            st.sidebar.write("")
            i+=1
        st.sidebar.markdown(f"""<h4>Stock Name: <span style="color:green">{user_input}</span></h4>""",unsafe_allow_html=True)
        st.sidebar.markdown(f"""<h4>Stock Symbol: <span style="color:green">{dict[user_input]}</span></h4>""",unsafe_allow_html=True)
        st.sidebar.write("Mean Squared Error:") 
        st.sidebar.write(mse)

        st.sidebar.write("Root Mean Squared Error:")
        st.sidebar.write(rmse)

        # st.sidebar.write("Accuracy:")
        # st.sidebar.write(accuracy)

        # st.sidebar.write("Mean Absolute Percentage Error:")
        # st.sidebar.write(mape)


        # Final Graph
        heading5 = f"""<p style="font-family:serif; font-size: 30px;">Predictions VS Original</p>"""
        st.markdown(heading5, unsafe_allow_html=True)
        st.spinner()
        with st.spinner(text='Loading the result...'):
            time.sleep(1)
        fig2=plt.figure(figsize=(12,6))
        plt.plot(y_test,'b',label='Original Price')
        plt.plot(y_predicted,'r',label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)
        # st.balloons()
    elif uploaded_file:
        df=pd.read_csv(uploaded_file)
        # Describing Data
        heading1 = f"""<p style="font-family:serif; font-size: 30px;">Data from {start}-{end}</p>"""
        st.markdown(heading1, unsafe_allow_html=True)
        st.write(df.describe())

        # Visualisation
        heading2 = '<p style="font-family:serif;  font-size: 30px;">Closing Price VS Time Chart</p>'
        st.markdown(heading2, unsafe_allow_html=True)
        fig=plt.figure(figsize=(12,6))
        plt.plot(df.Close,'b',label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        heading3 = '<p style="font-family:serif;font-size: 30px;">Closing Price VS Time Chart with 100MA</p>'
        st.markdown(heading3, unsafe_allow_html=True)
        ma100=df.Close.rolling(100).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r',label='100 days MA')
        plt.plot(df.Close,'b',label='Closing Price')
        plt.legend()
        st.pyplot(fig)

        heading4 = '<p style="font-family:serif; font-size: 30px;">Closing Price VS Time Chart with 100MA & 200MA</p>'
        st.markdown(heading4, unsafe_allow_html=True)
        ma100=df.Close.rolling(100).mean()
        ma200=df.Close.rolling(200).mean()
        fig=plt.figure(figsize=(12,6))
        plt.plot(ma100,'r',label='100 days MA')
        plt.plot(ma200,'g',label='200 days MA')
        plt.plot(df.Close,'b',label='Closing Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)


        # splitting data into training and testing

        data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler(feature_range=(0,1))

        data_training_array=scaler.fit_transform(data_training)

        st.spinner()
        with st.spinner(text='Loading LSTM model...'):
            time.sleep(1)
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
            time.sleep(2) 
        y_predicted=model.predict(x_test)
        scaler=scaler.scale_

        mse=np.square(np.subtract(y_test,y_predicted)).mean()
        rmse=np.sqrt(mse)
        # rmse=np.sqrt(np.mean(((y_predicted-y_test)**2)))

        accuracy=100-(rmse*100)

        mape=np.mean(np.abs((y_test-y_predicted)/y_test))*100

        scale_factor=1/scaler[0]
        y_predicted=y_predicted*scale_factor
        y_test=y_test*scale_factor

            

        i=1
        while i<5:
            st.sidebar.write("")
            i+=1
        st.sidebar.markdown(f"""<h4>Stock Symbol: <span style="color:green">{df.Symbol[0]}</span></h4>""",unsafe_allow_html=True)
        st.sidebar.write("Mean Squared Error:") 
        st.sidebar.write(mse)

        st.sidebar.write("Root Mean Squared Error:")
        st.sidebar.write(rmse)


        # st.sidebar.write("Accuracy:")
        # st.sidebar.write(accuracy)
        # st.sidebar.write("Mean Absolute Percentage Error:")
        # st.sidebar.write(mape)


        # Final Graph
        heading5 = f"""<p style="font-family:serif; font-size: 30px;">Predictions VS Original</p>"""
        st.markdown(heading5, unsafe_allow_html=True)
        st.spinner()
        with st.spinner(text='Loading the result...'):
            time.sleep(1)
        fig2=plt.figure(figsize=(12,6))
        plt.plot(y_test,'b',label='Original Price')
        plt.plot(y_predicted,'r',label='Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig2)
        # st.balloons()
    else:
        st.info('Please select a stock ticker or upload a csv file')




if rad=='Contact':
    i=1
    while i<14:
        st.sidebar.write("")
        i+=1
    st.sidebar.write("Visit Developer Profile:")
    st.sidebar.write("[Prasanna Kumar Baniya](https://www.linkedin.com/in/prasanna-kumar-baniya-9a91a5179/)")
    st.sidebar.write("[Sudhan Neupane](https://www.facebook.com/madhu.neupane.10)")
    st.sidebar.write("[Vikash Palli](https://www.facebook.com/vikashpalli.mgr)")
    new_title = '<p style="font-family:serif; font-size: 30px;"><b>We would love to hear from you!</b><br>Send us a message!</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    with st.form("my_form",clear_on_submit=True):
       name=st.text_input(label='Name:',max_chars=50,placeholder='Enter your name')    
       email=st.text_input(label='Email:',placeholder='Enter your email address')
       message=st.text_area(label='Message:',height=100)
    #    rating = st.slider("Rate us (1 to 5)",0,5,value=3)
    #    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
       submitted = st.form_submit_button("Submit")
    if submitted:
        # d = {"Name":name, 
        #     "Email":email,
        #     "Message":message,
        #     "Rating":rating,}
        d = {"Name":name, 
            "Email":email,
            "Message":message}

        if name=="" and email=="" and message=="":
            # st.markdown('<h5>Please fill all the fields<h5>',unsafe_allow_html=True)
            st.error('Please fill all the fields')
        else:
           st.markdown('<h3 style="font-family:sans-serif; color:#8271D2;">Thank you for your feedback!</h3>', unsafe_allow_html=True)  
           df=pd.read_json('response.json')
           df = df.append(d, ignore_index = True)
           open('response.json', 'w').write(df.to_json())



         

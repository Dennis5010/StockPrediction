from datetime import datetime
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = datetime.datetime.now()

st.title("STOCK TREND PREDICTION")

user_input = st.text_input("Enter Stock Symbol",'AAPL')

df = data.DataReader(user_input , 'yahoo' , start , end )

#Describing Data 

st.subheader('Data from 2010 - Till todays date')
st.write(df.describe())

#VISUALIZATION
st.subheader('Closing price vs Time Chart')
fig = plt.figure(figsize=(12,8))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time Chart with 42days Moving Average')
st.subheader('GREEN COLOR IS CLOSING PRICE & RED IS MA42')
ma42 = df.Close.rolling(42).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma42 , 'r')
plt.plot(df.Close,'g')
st.pyplot(fig)


st.subheader('Closing price vs Time Chart with 200days Moving Average')
st.subheader('GREEN COLOR IS CLOSING PRICE & RED IS MA42 & BLUE IS MA200')
ma42 = df.Close.rolling(42).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,8))
plt.plot(ma42 , 'r')
plt.plot(ma200 , 'b' )
plt.plot(df.Close , 'g')
st.pyplot(fig)


#Splitting the Data into Training and testing Data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_arr = scaler.fit_transform(data_training)

#load MY MODEL
model = load_model('C:\\Users\\denni\\OneDrive\\Desktop\\Data_science\\Stock_prediction\\kreas_model (1).h5')

#testing part
past_42_days = data_training.tail(42)
final_df=past_42_days.append(data_testing,ignore_index=True)
data_input = scaler.fit_transform(final_df)

x_test = []
y_test = []


for i in range(42,data_input.shape[0]):
  x_test.append(data_input[i-42:i])
  y_test.append(data_input[i,0])

x_test , y_test = np.array(x_test),np.array(y_test)
y_pred = model.predict(x_test)

scaler = scaler.scale_

scale_fact = 1/scaler[0]
y_pred = y_pred * scale_fact
y_test = y_test * scale_fact

#Final PREDICTED GRAPH

st.subheader('Predicted vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label= 'Orignal Price')
plt.plot(y_pred , 'r' , label = 'Predicted price')
plt.xlabel('TIME')
plt.ylabel('PRICE')
plt.legend()
st.pyplot(fig2)
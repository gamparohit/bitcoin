import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from prophet import Prophet
import datetime 
from sklearn.metrics import accuracy_score
#Fetching data for each day
oneday_data = pd.read_csv("BTC-USDT-1d.csv")
oneday_data = oneday_data[['open_time','close_time','open','high','low','close']]
#Summarizing the data
oneday_data.describe()


#EDA 

#Plotting open time and rate at each day
x = oneday_data['open_time']
y = oneday_data['open']
plt.plot(x,y,label='Price each day at open time')
plt.xlabel("Open time each day")
plt.ylabel("Open price each day")

#Plotting close time and rate at each day
x = oneday_data['close_time']
y = oneday_data['close']
plt.plot(x,y,label='Price each day at close time')
plt.xlabel("Close time each day")
plt.ylabel("Close price each day")

#Plotting open & close each day
open_close = oneday_data[['open','close']]
open_close[['open','close']].plot()

#Conclusion1 : There isn't significant difference in the open and close prices.We can use either open or close prices for further predictions

#highest vs open
x = oneday_data['open']
y1 = oneday_data['high']
x.plot()
y1.plot()

#Conclusion 2 : The open and close rates almost overlap each other

#checking if the price went high at closing on each day
oneday_data['open_close_diff'] = oneday_data['close'] - oneday_data['open']
oneday_data['isGain'] = oneday_data['open_close_diff'].apply(lambda x :'True' if x > 0 else 'False')

#Conclusion 3 : We observe that almost 53% of the data has a high close rate as compared to open rate

#checking avg highest it went daily from opening
oneday_data['open_high_diff'] = oneday_data['high'] - oneday_data['open']


#PREDCITIONS:

#Prediction using classification to check based on open rate if it the stock will end with a higher rate
x = oneday_data[['open','high','low']]
y = oneday_data[['isGain']]
x_train,x_test,y_train,y_test = train_test_split(x,y)
#Logistic Regression
model = LogisticRegression()
model.fit(x_train,y_train)
pred = model.predict(x_test)
accuracy_score(y_test, pred)


#Conclusion 4 : The logistic regression gives an accuracy of 87% 
y_test = y_test.reset_index(drop=True)
prediction_check = pd.DataFrame()
prediction_check['pred'] = pred
prediction_check['y_test'] = y_test

#KN neighbors
KNN_model = KNeighborsClassifier(n_neighbors=2)
KNN_model.fit(x_train, y_train)
KNN_prediction = KNN_model.predict(x_test)
accuracy_score(y_test, KNN_prediction)

#Conclusion 5 : The accuracy achieved was around 71%
#Hence we go ahead with implementing time series analysis using Prophet,in order to analyse the trend, and prediction.

#For more intricate details, we will dig deeper and try predicting the close prices on an hourly basis.

#Preprocessing the close time column

#Historical closing BTC prices in order to predict future BTC ones
hourly_data = pd.read_csv("BTC-USDT-1h.csv")
hourly_data = hourly_data[['close_time','close']]
hourly_data['close_time'] = hourly_data['close_time'].apply(lambda x : datetime.datetime.fromtimestamp(int(x)/1000))

#Plotting our hourly data
x = hourly_data['close_time']
y = hourly_data['close']
plt.plot(x,y,label='Price each hour at close time')
plt.xlabel("Close time each hour")
plt.ylabel("Close price each hour")
plt.title("Hourly data")

hourly_data.set_index('close_time',inplace = True)
ts = pd.DataFrame({'ds':hourly_data.index,'y':hourly_data.close})

#Splitting our data into training and testin
prophet_model = Prophet(changepoint_prior_scale=0.75,changepoint_range = 1)
#Changepoints are moments in the data where the data shifts direction _ default - 0.05, increasing
# it allows automatic detection of more change points
#Changepoint range - deafult = 0.8, making it 1 will incorporate all datapoints while identifying changepoints

prophet_model.fit(ts)

future = prophet_model.make_future_dataframe(periods=50)
forecast = prophet_model.predict(future)

# display the most critical output columns from the forecast
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
fig = prophet_model.plot(forecast)
prophet_model.plot_components(forecast)

#Predicting future close rates on daily records
daily_data = oneday_data[['close_time','close']]
daily_data['close_time'] = daily_data['close_time'].apply(lambda x : datetime.datetime.fromtimestamp(int(x)/1000))

#Plotting our hourly data
x = daily_data['close_time']
y = daily_data['close']
plt.plot(x,y,label='Price each day at close time')
plt.xlabel("Close time each day")
plt.ylabel("Close price each day")
plt.title("Daily data")

daily_data.set_index('close_time',inplace = True)
ts = pd.DataFrame({'ds':hourly_data.index,'y':hourly_data.close})

#Splitting our data into training and testin
prophet_model = Prophet(changepoint_prior_scale=0.75,changepoint_range = 1)
#Changepoints are moments in the data where the data shifts direction _ default - 0.05, increasing
# it allows automatic detection of more change points
#Changepoint range - deafult = 0.8, making it 1 will incorporate all datapoints while identifying changepoints

prophet_model.fit(ts)

future = prophet_model.make_future_dataframe(periods=50)
forecast = prophet_model.predict(future)

# display the most critical output columns from the forecast
forecast[['ds','yhat','yhat_lower','yhat_upper']].head()

# plot
fig = prophet_model.plot(forecast)
prophet_model.plot_components(forecast)

 


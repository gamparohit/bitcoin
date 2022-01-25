import requests as rq
import pandas as pd
# API to download the data from the website about bitcoin fear and greed index
Output = rq.get('https://api.alternative.me/fng/?limit=0').json()['data']
#Coverting the list of json into dataframe for better storage
Df = pd.DataFrame(Output)
#Converting the epoch time to datetime
Df['timestamp'] = pd.to_datetime(Df['timestamp'],unit='s')
#Converting the dataframe to csv
Df.to_csv('D:/Coding Projects/project-rgampa-mtavilda/Data/bitcoin_fear_and_greed_index.csv',index = False)
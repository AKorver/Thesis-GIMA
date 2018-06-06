import pandas as pd
import os
import glob
import numpy as np

path = "C:/Users/Aaron Korver/Desktop/analyse2/Analyse B-Riders kenmerken/joinedtracks/"
allfiles = glob.glob(os.path.join(path,"*.csv"))
df3 = pd.DataFrame(columns=['track','datetime','person','Starttijd','Eindtijd','tijdverschil','Purpose'])
for file_ in allfiles:
    df1 = pd.read_csv(file_)
    df2 = df1[['track','datetime','person','purpose']]
    routeidlist = df1.track.unique()
    for value in routeidlist:
        dfTemp = df2.loc[df2['track'] == value]
        dfTemp['Starttijd'] = dfTemp['datetime'].iloc[0]
        dfTemp['Eindtijd'] = dfTemp['datetime'].iloc[-1]
        dfTemp['tijdverschil'] = (pd.to_datetime(dfTemp['datetime'].iloc[-1], format='%Y-%m-%d %H:%M:%S') - pd.to_datetime(dfTemp['datetime'].iloc[0], format='%Y-%m-%d %H:%M:%S')).total_seconds() / 60.0
        dfTemp['Purpose'] = dfTemp['purpose'].iloc[-1]
        df3 = df3.append(dfTemp.iloc[0])
        print df3['Purpose']
    del df1
    del df2
df3.to_csv("C:/Users/Aaron Korver/Desktop/tijdenpurpose.csv")

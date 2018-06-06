import pandas as pd
import numpy as np

#Imports CSV
df1 = pd.read_csv("C:\Users\Aaron Korver\Desktop\outputvooranalyse.csv")

#Formations according to variable scheme
df1['WEGTYPE'] = df1['WEGTYPE'].replace(['normale weg', 'ventweg', 'bromfietspad (langs weg)','solitair bromfietspad','voetgangersdoorsteekje','veerpont','voetgangersgebied'], '1')
df1['WEGTYPE'] = df1['WEGTYPE'].replace(['weg met fiets(suggestie)strook', 'fietspad (langs weg)', 'solitair fietspad', 'fietsstraat'], '2')
df1['WEGTYPE'] = df1['WEGTYPE'].replace(['ONBEKEND', ' '], np.nan)

df1['WEGKWAL'] = df1['WEGKWAL'].replace(['goed'], '3')
df1['WEGKWAL'] = df1['WEGKWAL'].replace(['redelijk'], '2')
df1['WEGKWAL'] = df1['WEGKWAL'].replace(['slecht'], '1')
df1['WEGKWAL'] = df1['WEGKWAL'].replace(['ONBEKEND', ' '], np.nan)

df1['VERLICHTIN'] = df1['VERLICHTIN'].replace(['goed verlicht'], '3')
df1['VERLICHTIN'] = df1['VERLICHTIN'].replace(['beperkt verlicht (bijvoorbeeld alleen bij kruispunten)'], '2')
df1['VERLICHTIN'] = df1['VERLICHTIN'].replace(['niet verlicht'], '1')
df1['VERLICHTIN'] = df1['VERLICHTIN'].replace(['ONBEKEND', ' '], np.nan)

df1['OMGEVING'] = df1['OMGEVING'].replace(['bos', 'natuur (behalve bos)'], '3')
df1['OMGEVING'] = df1['OMGEVING'].replace(['bebouwd (veel groen)', 'landelijke of dorps', 'akkers/weilanden'], '2')
df1['OMGEVING'] = df1['OMGEVING'].replace(['bebouwd (weinig of geen groen)'], '1')
df1['OMGEVING'] = df1['OMGEVING'].replace(['ONBEKEND', ' '], np.nan)

df1['SCHOONHEID'] = df1['SCHOONHEID'].replace(['mooi', 'schilderachtig'], '4')
df1['SCHOONHEID'] = df1['SCHOONHEID'].replace(['neutraal'], '3')
df1['SCHOONHEID'] = df1['SCHOONHEID'].replace(['lelijk/saai'], '2')
df1['SCHOONHEID'] = df1['SCHOONHEID'].replace(['zeer lelijk'], '1')
df1['SCHOONHEID'] = df1['SCHOONHEID'].replace(['ONBEKEND', ' '], np.nan)

df1['MAXSNELHEI'] = df1['MAXSNELHEI'].replace(['stapvoets (15)'], 15)
df1['MAXSNELHEI'] = df1['MAXSNELHEI'].replace(['ONBEKEND', ' '], 0)

count = 1
#Drops duplicates
dfUnique = df1['ROUTEID'].drop_duplicates()
#Generates output DF
df5 = pd.DataFrame(columns=['routeid','id','avmaxspeed','avtripspeed','routelength','frtypeofroad','avtrafficvolume','avomgevingsvar','avschoonheidsvar'])

#Iterates over rows and calculates unique values per routeid
for row in dfUnique:
    df2 = df1.loc[df1['ROUTEID'] == row]
    df2['maxspeedxroutelength'] = df2['MAXSNELHEI'].astype(float) * df2['SHAPE_LENGTH']
    df2['avmaxspeed'] = df2['maxspeedxroutelength'].sum() / df2['SHAPE_LENGTH'].sum()
    
    df2['avtripspeedCalc'] = (df2['SHAPE_LENGTH'] / df2['SHAPE_LENGTH'].sum())*df2['SNELHEID']
    df2['avtripspeed'] = df2['avtripspeedCalc'].sum()
    
    df2['routelength'] = df2['SHAPE_LENGTH'].sum()
    
    df2['frtypeofroadCalc'] = (df2['SHAPE_LENGTH'] / df2['SHAPE_LENGTH'].sum())*df2['WEGTYPE'].astype(float)
    df2['frtypeofroad'] = df2['frtypeofroadCalc'].sum()
    
    df2['avtrafficvolumeCalc'] = (df2['SHAPE_LENGTH'] / df2['SHAPE_LENGTH'].sum())*df2['INTENSITEI']
    df2['avtrafficvolume'] = df2['avtrafficvolumeCalc'].sum()
    
    df2['avomgevingsvarCalc'] = (df2['SHAPE_LENGTH'] / df2['SHAPE_LENGTH'].sum())*df2['OMGEVING'].astype(float)
    df2['avomgevingsvar'] = df2['avomgevingsvarCalc'].sum()
    
    df2['avschoonheidsvarCalc'] = (df2['SHAPE_LENGTH'] / df2['SHAPE_LENGTH'].sum())*df2['SCHOONHEID'].astype(float)
    df2['avschoonheidsvar'] =  df2['avschoonheidsvarCalc'].sum()
    
    df3 = pd.DataFrame(columns=['routeid','id','avmaxspeed','avtripspeed','routelength','frtypeofroad','avtrafficvolume','avomgevingsvar','avschoonheidsvar'])
    df3['routeid'] = df2['ROUTEID']
    df3['id'] = df2['ID']
    df3['avmaxspeed'] = df2['avmaxspeed']
    df3['avtripspeed'] = df2['avtripspeed']
    df3['routelength'] = df2['routelength']
    df3['frtypeofroad'] = df2['frtypeofroad']
    df3['avtrafficvolume'] = df2['avtrafficvolume']
    df3['avomgevingsvar'] = df2['avomgevingsvar']
    df3['avschoonheidsvar'] = df2['avschoonheidsvar']
    df4 = df3.iloc[0:1]
    df5 = df5.append(df4)
    print "Rendered ROUTEID #" + str(count)
    count = count + 1
    del df2,df3,df4
#Generates output CSV
df5.to_csv("C:\Users\Aaron Korver\Desktop\outputFinal.csv")

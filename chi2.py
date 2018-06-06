import pandas as pd
import numpy as np
import scipy
from scipy.stats.contingency import chi2_contingency

df1 = pd.read_csv("C:/Users/Aaron Korver/Desktop/AnalyseCijfers2.csv")
df1['trip_purpose'] = df1['trip_purpose'].replace(['betaaldwerk', 'dagelijkeboodschappen', 'diensten','niet-dagelijkeboodschappen','studie'], '1')
df1['trip_purpose'] = df1['trip_purpose'].replace(['recreatie', 'sociaal', 'vrijetijd','home'], '0')

contingency = pd.crosstab(df1['BG2010NameDest'], df1['trip_purpose'])

contingency.to_csv("C:/Users/Aaron Korver/Desktop/ChiSquareOrig3.csv")
chi2, p, dof, expected = chi2_contingency(contingency)
print chi2, p, dof

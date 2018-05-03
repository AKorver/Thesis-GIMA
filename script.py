import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Loads data as Pandas dataframe
data = pd.read_csv("C:/Users/Aaron Korver/Desktop/AnalyseCijfers2.csv")

# Converts motives to utilitarian or non-utilitarian
data['trip_purpose'] = data['trip_purpose'].replace(['betaaldwerk', 'dagelijkeboodschappen', 'diensten','niet-dagelijkeboodschappen','studie'], '1')
data['trip_purpose'] = data['trip_purpose'].replace(['recreatie', 'sociaal', 'vrijetijd','home'], '0')

# Create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('trip_purpose ~ C(trackDest)+ C(BG2010AOrigin) + avg_trip_speed + avg_max_speed + \
                 route_length +  avg_traffic_volume + fraction_type_of_road + avg_environment_value + avg_appreciation_environment ',
                  data, return_type="dataframe")

# Print y
y.drop(y.columns[0], axis = 1) 

# Converts y dataframe to 1d numpy array
y = np.ravel(y['trip_purpose[1]'])

# Define names of classifiers
names = ["Logistic Regression","Decision Tree", "Random Forest", "SVM", "K-Nearest Neighbors",
             "Naive Bayes", "Neural Network", "Gradient Boosting Method"]
# Define SKLearn classifier methods
classifiers = [
        LogisticRegression(),
        tree.DecisionTreeClassifier(criterion='gini'),
        RandomForestClassifier(),
        svm.SVC(),
        KNeighborsClassifier(),
        GaussianNB(),
        MLPClassifier(alpha=1),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)]

# Define K-Fold Crossvalidation K-value
crossfreq = 10

#Print Variables that are used
print("Full Model")

# Loops trough classifiers, fits the model and calculates accurac, precision, recall and F1
for name, clf in zip(names,classifiers):
    model = clf.fit(X, y)
    model.score(X, y)
    print(name + " Model Score: " + str(model.score(X, y)))
    scoresAcc = cross_val_score(model, X, y, scoring='accuracy', cv=crossfreq)
    scoresAvp = cross_val_score(model, X, y, scoring='average_precision', cv=crossfreq)
    scoresRec = cross_val_score(model, X, y, scoring='recall', cv=crossfreq)
    scoresF1 = cross_val_score(model, X, y, scoring='f1', cv=crossfreq)
    print ("10 fold crossvalidated accuracy score for " + name + ": " + str(scoresAcc.mean()))
    print ("10 fold crossvalidated average precision score for " + name + ": " + str(scoresAvp.mean()))
    print ("10 fold crossvalidated recall score for " + name + ": " + str(scoresRec.mean()))
    print ("------------------------------------------------------------------")
    del model,scoresAcc,scoresAvp,scoresRec,scoresF1
    
# Model score for KMeans is calculated and printed seperately bc the nature of the method does not support precision/recall
model = KMeans(n_clusters=2, random_state=0).fit(X, y)
model.score(X, y)
scoresKMeans = cross_val_score(KMeans(n_clusters=2, random_state=0), X, y, scoring='accuracy', cv=10)
print("K-Means Model Score: " + str(model.score(X, y)))
print("10 fold crossvalidated accuracy score for K-Means" + str(scoresKMeans.mean()))
print ("------------------------------------------------------------------")

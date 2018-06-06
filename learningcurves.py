import matplotlib.pyplot as plt
import pandas as pd
from patsy import dmatrices
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of training examples")
    plt.ylabel("Accuracy score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation accuracy (K=10)")

    plt.legend(loc="best")
    return plt

# Loads data as Pandas dataframe
data = pd.read_csv("C:/Users/Aaron Korver/Desktop/AnalyseCijfers2.csv")

# Converts motives to utilitarian or non-utilitarian
data['trip_purpose'] = data['trip_purpose'].replace(['betaaldwerk', 'dagelijkeboodschappen', 'diensten','niet-dagelijkeboodschappen','studie'], '1')
data['trip_purpose'] = data['trip_purpose'].replace(['recreatie', 'sociaal', 'vrijetijd','home'], '0')

# Create dataframes with an intercept column and dummy variables for
# occupation and occupation_husb
y, X = dmatrices('trip_purpose ~ C(trackDest)+ C(BG2010AOrigin)+ avg_trip_speed + avg_max_speed + \
                 route_length +  avg_traffic_volume + fraction_type_of_road + avg_environment_value + avg_appreciation_environment',
                  data, return_type="dataframe")
# Print y
y.drop(y.columns[0], axis = 1) 

# Converts y dataframe to 1d numpy array
y = np.ravel(y['trip_purpose[1]'])
print("klaar")


title = "Learning Curves (Naive Bayes)"
estimator = GaussianNB()
print("klaar")
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

print("klaar")
title = "Learning Curves (Logistic Regression)"
estimator = LogisticRegression()
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (Decision Tree)"
estimator = tree.DecisionTreeClassifier(criterion='gini')
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (Random Forest)"
estimator = RandomForestClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (Support Vector Machine)"
estimator = svm.SVC()
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (K-nearest Neighbor)"
estimator = KNeighborsClassifier()
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (Neural Network)"
estimator = MLPClassifier(alpha=1)
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (K-Means)"
estimator = KMeans(n_clusters=3, random_state=0)
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)

title = "Learning Curves (Gradient Boosting (GBM)"
estimator = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
plot_learning_curve(estimator, title, X, y, ylim=(0, 1.01), cv=5, n_jobs=1)


plt.show()

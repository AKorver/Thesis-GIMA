import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

df1 = pd.read_csv("C:/Users/Aaron Korver/Desktop/analyseCijfers2.csv")
df1 = df1[df1.id != 124946]
plt.style.use('ggplot')
#plotlist = ["Trip duration","Average Max Speed","Average Trip Speed","Route length","Fraction type of road","Average Traffic Volume","Average Environment Value","Average Appreciation Environment"]
plotlist = [""]
for values in plotlist:
    df2 = df1[[values,'Trip purpose']]
    df2 = df2[np.isfinite(df2[values])]
    groups = df2.groupby("Trip purpose").groups
    vrijetijd = df2[values][groups["vrijetijd"]]
    betaaldwerk = df2[values][groups["betaaldwerk"]]
    dagelijkeboodschappen = df2[values][groups["dagelijkeboodschappen"]]
    diensten = df2[values][groups["diensten"]]
    home = df2[values][groups["home"]]
    nietdagelijkeboodschappen = df2[values][groups["niet-dagelijkeboodschappen"]]
    recreatie = df2[values][groups["recreatie"]]
    sociaal = df2[values][groups["sociaal"]]
    studie = df2[values][groups["studie"]]
    print("------------------------")
    print(str(values) + " for different trip purposes")
    print("------------------------")
    print("vrije tijd standaarddeviatie: " + str(np.std(vrijetijd)) + "; vrije tijd gemiddelde: " + str(np.mean(vrijetijd)))
    print("betaaldwerk standaarddeviatie: " + str(np.std(betaaldwerk)) + "; betaaldwerk gemiddelde: " + str(np.mean(betaaldwerk)))
    print("dagelijkeboodschappen standaarddeviatie: " + str(np.std(dagelijkeboodschappen)) + "; dagelijkeboodschappen gemiddelde: " + str(np.mean(dagelijkeboodschappen)))
    print("diensten standaarddeviatie: " + str(np.std(diensten)) + "; diensten gemiddelde: " + str(np.mean(diensten)))
    print("home standaarddeviatie: " + str(np.std(home)) + "; home gemiddelde: " + str(np.mean(home)))
    print("nietdagelijkeboodschappen: " + str(np.std(nietdagelijkeboodschappen)) + "; nietdagelijkeboodschappen: " + str(np.mean(nietdagelijkeboodschappen)))
    print("recreatie: " + str(np.std(recreatie)) + "; recreatie: " + str(np.mean(recreatie)))
    print("sociaal standaarddeviatie: " + str(np.std(sociaal)) + "; sociaal gemiddelde: " + str(np.mean(sociaal)))
    print("studie standaarddeviatie: " + str(np.std(studie)) + "; studie: " + str(np.mean(studie)))
    print("------------------------")
    print stats.f_oneway(vrijetijd, betaaldwerk, dagelijkeboodschappen, diensten, home, nietdagelijkeboodschappen, recreatie, sociaal, studie)
    print("------------------------")
    
    del df2, groups, vrijetijd, betaaldwerk, dagelijkeboodschappen, diensten, home, nietdagelijkeboodschappen, recreatie, sociaal, studie
    #fig = plt.figure(1, figsize=(12, 6))
    df1.boxplot(values, by='Trip purpose')
    plt.xlabel('Trip Purpose', fontsize=20)
    plt.ylabel(values, fontsize=20)
    plt.xticks(fontsize=12)
    plt.autoscale(enable=True, axis='y', tight=False)
    plt.title('Boxplot of '+ values + " for different trip purposes", fontsize=28)
    plt.show()
    plt.savefig("C:/Users/Aaron Korver/Desktop/output/" + values + ".png")
    plt.gcf().clear()
#df3.to_csv("C:/Users/Aaron Korver/Desktop/analyseCijfers.csv")

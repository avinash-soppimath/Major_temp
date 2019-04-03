#from __future__ import division, print_function
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import csv
import warnings
warnings.filterwarnings("ignore")
#from scipy import stats

#Training the dataset 
dataset_train = pd.read_csv("Traindata.csv")
#dataset_test = pd.read_csv("Testdata.csv")
#print(dataset.shape)

#Encoding categorical data
data1 = dataset_train[['type','temp','rainfall','humidity','n','p','k','season','soil']]
dataset1 = pd.get_dummies(data1)
#print(dataset2.iloc[:,:].head(5))
#print(data2.head())
#print(dataset2.shape)

X = dataset1.iloc[:,1:14].values
Y = dataset1.iloc[:,0].values

#Splitting dataset into training and testing dataset
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.15,random_state=0)
#test_size=0.16
#Feature scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
 
clf= SVC(kernel = 'rbf', degree = 2, gamma='auto', random_state=0, C=2)
clf.fit(X_train,Y_train)

#Predicting test dataset results
Y_pred = clf.predict(X_test)
#final_pred = np.array([])
#for i in range(0,len(X_test)):
#    final_pred = np.append(final_pred, np.average(Y_pred2,Y_pred3,Y_pred4,Y_pred5))
#Y_pred = np.average(Y_pred1,Y_pred2,Y_pred3,Y_pred4,Y_pred5,Y_pred6)
#print(dataset_train.crop[Y_pred-1])
#Print accuracy
print("Training dataset size: ", len(X_train))
print("Test dataset size: ", len(X_test))
print("Accuracy: ",accuracy_score(Y_test,Y_pred))
#print("Accuracy: ",np.average(accuracy_score(Y_test,Y_pred2),accuracy_score(Y_test,Y_pred3),accuracy_score(Y_test,Y_pred4),accuracy_score(Y_test,Y_pred5)))

"""
#for reading from csv file and predicting
dataset_output = pd.read_csv("Output.csv")
X_output = dataset_output.iloc[:,0:13].values
data3 = dataset_output[['temp','rainfall','humidity','n','p','k','season','soil']]
X_output = pd.get_dummies(data3)
X_output = X_output.iloc[:,:].values
X_output = X_output[:,0:13]
Y_output = clf.predict(X_output)
print(dataset_train.crop[Y_output-1])
"""

temp = float(input("Enter the average temperature in celcius: "))
rainfall = float(input("Enter the rainfall in mm in your region: "))
humidity = float(input("Enter humidity: "))
n = float(input("Enter the nitrogen content in kg/ha: "))
p = float(input("Enter the phosphorous content in kg/ha: "))
k = float(input("Enter the potassium content in kg/ha: "))
season = int(input("Enter the season, 0 for kharif, 1 for rabi: "))
soiltype = int(input(('Enter the soil type: 0 for alluvial soil, 1 for black soil,2 for laterite soil,3 for marshy soil,4 for red soil: ') ))
print("Crop suitable for growing:: ")
if (season == 0):
    if(soiltype == 0):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,1,0,1,0,0,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'alluvial soil'
    elif(soiltype == 1):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,1,0,0,1,0,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'black soil'
    elif(soiltype == 2):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,1,0,0,0,1,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'laterite soil'
    elif(soiltype == 3):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,1,0,0,0,0,1,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'marshy soil'
    elif(soiltype == 4):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,1,0,0,0,0,0,1]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'red soil'
    season_name = 'kharif'
else:
    if(soiltype == 0):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,0,1,1,0,0,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'alluvial soil'
    elif(soiltype == 1):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,0,1,0,1,0,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'black soil'
    elif(soiltype == 2):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,0,1,0,0,1,0,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'laterite soil'
    elif(soiltype == 3):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,0,1,0,0,0,1,0]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'marshy soil'
    elif(soiltype == 4):
        p = clf.predict([[temp,rainfall,humidity,n,p,k,0,1,0,0,0,0,1]])
        q = dataset_train.crop[p[0]-1]
        soil_name = 'red soil'
    season_name = 'rabi'
print(q)
z = int(input("Enter 1 if predicted crop is suitable, else enter 0: "))
if z == 0:
    print("1-maize\n2-cotton\n3-wheat\n4-paddy\n5-barley\n6-finger millet\n7-ground nut\n8-green gram\n9-tea\n10-chillies\n11-turmeric\n12-sugar cane\n13-jute\n14-sorghum\n15-bajra\n16-red gram\n17-black gram\n18-sesame\n19-sweet potato\n20-mustard\n21-soyabean\n22-lin seed\n23-niger\n24-capsicum\n25-cucumber\n26-ridge gourd\n27-bottle gourd\n28-snake gourd\n29-Water melon\n30-muskmelon\n31-cauli flower\n32-onion\n33-french bean\n34-garden pea\n")
    p[0] = int(input("Enter the crop you prefer to grow: "))
    q = dataset_train.crop[p[0]-1]
row = [q, temp, rainfall, humidity, n, p, k, season_name, soil_name, p[0]]
with open('Traindata.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)
csvFile.close()

plt.plot(X,'ro')
plt.plot(X_train,'bo')
plt.plot(X_test,'go')
plt.show() 

plt.plot(Y,'ro')
plt.plot(Y_train,'bo')
plt.plot(Y_test,'go')
plt.show()

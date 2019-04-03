#Importing libraries
import numpy as np
import pandas as pd
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

#Training the dataset 
dataset_train = pd.read_csv("Traindata.csv")
dataset_test = pd.read_csv("Testdata.csv")
#print(dataset.shape)

#Encoding categorical data
data1 = dataset_train[['type','temp','rainfall','humidity','n','p','k','season','soil']]
dataset2 = pd.get_dummies(data1)
#print(dataset2.iloc[:,:].head(5))
#print(data2.head())
#print(dataset2.shape)

X = dataset2.iloc[:,1:14].values
Y = dataset2.iloc[:,0].values

#Splitting dataset into training and testing dataset
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0)
X_test = dataset_test.iloc[:,0:14].values
data2 = dataset_test[['type','temp','rainfall','humidity','n','p','k','season','soil']]
X_test = pd.get_dummies(data2)
X_test = X_test.iloc[:,:].values
np.random.shuffle(X_test)
Y_test = X_test[:,0]
X_test = X_test[:,1:14]

#Fitting RandomForestClassifier to training dataset
#clf = RandomForestClassifier(n_estimators=100)
#clf.fit(X_train,Y_train) 
clf = SVC(kernel='rbf', degree = 3, gamma = 0.1)
clf.fit(X_train,Y_train)

#Predicting test dataset results
Y_pred = clf.predict(X_test)
print(dataset_train.crop[Y_pred-1])
#Print accuracy
#print("Accuracy: ",accuracy_score(Y_test,Y_pred))

"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""
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

"""
temp = int(input("Enter the average temperature in celcius: "))
rainfall = int(input("Enter the rainfall in mm in your region: "))
nitrogen_nutri = int(input("Enter the nitrogen content in kg/ha: "))
phosphor_nutri = int(input("Enter the phosphorous content in kg/ha: "))
potas_nutri = int(input("Enter the potassium content in kg/ha: "))
season = int(input("Enter the season, 0 for kharif, 1 for rabi,2 for yearlong season: "))
soiltype = int(input(('Enter the soil type: 0 for alluvial soil, 1 for black soil,2 for laterite soil,3 for marshy soil,4 for red soil: ') ))
print("Crop suitable for growing:: ")
if (season==0):
    if(soiltype==0):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,1,0,0,0,0]])-1])
    elif(soiltype==1):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,1,0,0,0]])-1]) 
    elif(soiltype==2):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,1,0,0]])-1])
    elif(soiltype==3):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,0,1,0]])-1])
    elif(soiltype==4):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,1,0,0,0,0,0,1]])-1])
else:
    if(soiltype==0):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,1,0,0,0,0]])-1])
    elif(soiltype==1):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,1,0,0,0]])-1])
    elif(soiltype==2):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,1,0,0]])-1])
    elif(soiltype==3):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,0,1,0]])-1])
    elif(soiltype==4):
                print(dataset.crop[clf.predict([[temp,rainfall,nitrogen_nutri,phosphor_nutri,potas_nutri,0,1,0,0,0,0,1]])-1])
"""

#graph
"""for i in range(0,len(X)):
    plt.plot(X[i][1],'ro')
    plt.plot(X[i][2],'bo')
    plt.plot(X[i][3],'go')
    plt.plot(X[i][4],'r--')
    plt.plot(X[i][5],'b--')"""
    
#Plotting the graph for X and Y values
plt.plot(X,'ro')

plt.plot(X_train,'bo')
plt.plot(X_test,'go')
#plt.plot(X, Y, color = 'blue')
plt.show() 

plt.plot(Y,'ro')
plt.plot(Y_train,'bo')
plt.plot(Y_test,'go')
#plt.plot(X, Y, color = 'blue')
plt.show() 

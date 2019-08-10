import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


dataset = pd.read_csv('data.csv')
print(dataset.head(5))
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# extract the input features and response variables
X= dataset.iloc[:,0:21]
y= dataset.iloc[:,21]

#normalize the input feature
sc= StandardScaler()
X = sc.fit_transform(X)
X
#split into training and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

#network architecture
classifier = Sequential()
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal', input_dim=21))
classifier.add(Dense(2, activation='relu', kernel_initializer='random_normal'))
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fit data to training
classifier.fit(X_train,y_train, batch_size=30, epochs= 15)


model_evaltn=classifier.evaluate(X_train, y_train)
print(model_evaltn)

#prediction
y_pred=classifier.predict(X_test)
#print('This is y_pred values:',y_pred)
y_pred =(y_pred>0.3)

#evaluate accuracy on test dataset
print('Confusion matrix:')
con_matrix = confusion_matrix(y_test, y_pred)
print( con_matrix)
print('Test accuracy: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
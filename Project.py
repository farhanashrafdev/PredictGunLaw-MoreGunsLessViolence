#!/usr/bin/env python

__author__      = " Farhan Ashraf & Awais Saleem "
__copyright__   = " Copyright 2021, Planet Earth "

# This is to import Numpy library
import numpy as np
# This is to import the pandas library
import pandas as pd

# Import writer class from csv module
from csv import writer


print(" \t  \t \tWelcome to Our Guns Law Pridictor")
print(" \t \t \tThis Program is made by Awais Saleem & Farhan Ashraf")
print("\t \t \tEnter Data to get Pridicition")


id_user = 1174
year = input("Enter Year in Number:")
violent = input("Enter Violent rate:")
murder = input("Enter murder rate:")
robbery = input("Enter number of robberies:")
prisoners = input("Enter number of prisoners:")
afam = input("Enter number of afam:")
cauc = input("Enter number of cauc:")
male_rate = input("Enter number of male_rate:")
population = input("Enter number of population:")
income = input("Enter number of income:")
density = input("Enter number of density:")
State = input("Enter name of State:")
law = 'yes'
myList = [id_user, year, violent, murder, robbery, prisoners, afam, cauc, male_rate, population, income, density, State,
          law]

with open('Guns.csv', 'a') as f_object:
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(myList)

    # Close the file object
    f_object.close()

    # This is to load the dataset that we will be performing data cleaning and classification on.
    df = pd.read_csv("Guns.csv")

    # this is to check the shape of the data and the data types that each attribute has.
#print(df.shape)
#print(df.dtypes)
# Basically This code below is used to confirm which of the columns are numbers
df_number = df.select_dtypes(include=[np.number])
number_columns = df_number.columns.values
#print(number_columns)
# This code below is to confirm which of the code below is not number data like text/string
df_not_number = df.select_dtypes(exclude=[np.number])
not_number_columns = df_not_number.columns.values
#print(not_number_columns)
# This is to find missing values in attributes, like null values.
for col in df.columns:
    missing_values_percentage = np.mean(df[col].isnull())
    #print('{} - {} Percentage'.format(col,round(missing_values_percentage*100)))

df.isnull().sum()
# To fix the problem of missing values, there are 3 ways according to the web: 1) Drop the row, 2) Drop the Column 3) Impute the missing
# We chose to impute the missing values with the mean

# df_numeric = df.select_dtypes(include=[np.number])
# numeric_cols = df_numeric.columns.values

#for col in numeric_cols:
 #   total_nullValues = df[col].isnull()
 #   num_missing = np.sum(total_nullValues)

  #  if num_missing > 0:  # for columns that have missing values
  #      mean = df[col].mean()
 #       df[col] = df[col].fillna(mean)

#######################################################################
# If we want to put the mean in all the values we can do using f fill also another way
df = df.interpolate()
df = df.fillna(method='ffill')

#######################################################################
# If we want to put the mean in all the values we can do

for col in df.columns:
    missing_values_percentage = np.mean(df[col].isnull())
    #print('{} - {} Percentage'.format(col,round(missing_values_percentage*100)))
#df.info()
#df.isna()

# Capitalization
df['state'].value_counts(dropna=False)
# To Fix the capitalization of the state attribute we use this.
# This function is basically putting the state.lower() column to be equal to the state column in our data set
df['state'] = df['state'].str.lower()
df['state'].value_counts(dropna=False)
# Cleaning Categorial Values we use these steps
df["law"] = df["law"].astype('category')
# this function basically converts the categorical values into 0,1
df["law"] = df['law'].cat.codes
#df.head()
df = pd.get_dummies(df,columns=['state'], prefix="State_")
df.to_csv (r'CleanedDataSet.csv', index = False, header=True)

dataset = pd.read_csv('CleanedDataSet.csv')
prediction_dataset = pd.DataFrame(dataset.iloc[-1:,:].values)
varForModelTest = np.delete(prediction_dataset.values,12,axis=1)
prediction_dataset = varForModelTest
#print(prediction_dataset)
dataset = dataset[:-1]

# This is the splitting our data set
from sklearn.model_selection import train_test_split

x = dataset.values
y = dataset['law'].values

x = np.delete(x, 12, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn import tree
decisionTree = tree.DecisionTreeClassifier(max_depth=5)
decisionTree.fit(x_train,y_train)   # This is for training the model
decisionTree.score(x_test,y_test)   # Predict the testing data

y_prediction = decisionTree.predict(prediction_dataset)
dT_accuracy = decisionTree.score(x_test,y_test)







# This is random forest classifier
from sklearn import ensemble

randomForestClassifier = ensemble.RandomForestClassifier(n_estimators=100)
randomForestClassifier.fit(x_train,y_train)
rForest_accuracy = randomForestClassifier.score(x_test,y_test)

# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
NB_Classifier = GaussianNB()
NB_Classifier.fit(x_train,y_train)
nb_accuracy = NB_Classifier.score(x_test,y_test)
# K-Nearest Neighbors Classification
from sklearn.neighbors import KNeighborsClassifier
KNN_Classifier = KNeighborsClassifier(n_neighbors=3)
KNN_Classifier.fit(x_train,y_train)
KNN_Classifier.score(x_test,y_test)

knn_accuracy = KNN_Classifier.score(x_test,y_test)






if dT_accuracy > rForest_accuracy and nb_accuracy and knn_accuracy:
    y_prediction = decisionTree.predict(prediction_dataset)
    print("The Accuracy of Decision Tree is: "+ dT_accuracy)

elif rForest_accuracy >  dT_accuracy and nb_accuracy and knn_accuracy:
    y_prediction = randomForestClassifier.predict(prediction_dataset)
    print("The Accuracy of Decision Tree is: ",rForest_accuracy )
elif nb_accuracy >  dT_accuracy and rForest_accuracy and knn_accuracy:
    y_prediction = NB_Classifier.predict(prediction_dataset)
    print("The Accuracy of Decision Tree is: ",nb_accuracy )
else:
    y_prediction = KNN_Classifier.predict(prediction_dataset)
    print("The Accuracy of Decision Tree is: ", y_prediction )

if y_prediction == [0]:
    print('Gun Law is NO')

else:
    print('Gun Law is YES')

print("\t \t \t EAST OR WEST SIR MANAN IS THE BEST :) :) :) ")

helo=input("THE END")
#!/usr/bin/env python
# coding: utf-8

# LOAD PACKAGES

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# READ DATA

# In[56]:


df = pd.read_csv('C:/Users/Muthiah/Downloads/heart_statlog_cleveland_hungary_final.csv')


# In[57]:


df.head()


# In[58]:


df.tail()


# DATA EXPLORATION

# In[59]:


df.info()


# In[60]:


df.isnull().sum()


# In[61]:


df.describe().T


# In[62]:


df['target'].value_counts()


# In[63]:


sns.countplot(x="target", data=df, palette="coolwarm")
plt.show()


# In[64]:


df['sex'].value_counts()


# In[65]:


sns.countplot(x='sex', data=df, palette="bwr")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()


# In[66]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()


# In[67]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Gender')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["NO Heart Disease found", "Heart Disease found"])
plt.ylabel('Frequency')
plt.show()


# In[68]:


plt.scatter(x=df.age[df.target==1], y=df.cholesterol[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.cholesterol[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


# In[73]:


pd.crosstab(df.fastingbloodsugar ,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["NO Heart Disease found", "Heart Disease found"])
plt.ylabel('Frequency')
plt.show()


# In[74]:


pd.crosstab(df.chestpaintype,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()


# SPLITTING FEATURES AND TARGET

# In[75]:


X = df.drop(columns='target',axis = 1)
Y = df['target']


# In[76]:


print(X)


# In[77]:


print(Y)


# SPLITTING DATA INTO TRAINING AND TEST DATA

# In[78]:


X_train,X_test,Y_train,Y_split = train_test_split(X,Y, test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape, X_test.shape)


# MODEL TRAINING

# LOGISTIC REGRESSION

# In[79]:


accuracies = {}
lr = LogisticRegression()
lr.fit(X_train,Y_train)

acc = lr.score(X_train, Y_train)*100
accuracies['LOGISTIC REGRESSION'] = acc
print("LOGISTIC REGRESSION Accuracy Score : {:.2f}%".format(acc))


# RANDOM FOREST

# In[80]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

acc = rf.score(X_train, Y_train)*100
accuracies['RANDOM FOREST'] = acc
print("RANDOM FOREST Algorithm Accuracy Score : {:.2f}%".format(acc))


# KNN

# In[81]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()  
knn.fit(X_train, Y_train)

acc = knn.score(X_train, Y_train)*100
accuracies['KNN'] = acc
print("KNN Algorithm Accuracy Score : {:.2f}%".format(acc))



# SVM

# In[82]:


from sklearn.svm import SVC  
svm = SVC()
svm.fit(X_train, Y_train)

acc = svm.score(X_train,Y_train)*100
accuracies['SVM'] = acc
print("Train Accuracy of SVM Algorithm: {:.2f}%".format(acc))



# NAIVE BAYES

# In[83]:


from sklearn.naive_bayes import GaussianNB  
nb = GaussianNB()  
nb.fit(X_train, Y_train)  

acc = nb.score(X_train, Y_train)*100
accuracies['NAIVE BAYES'] = acc
print("NAIVE BAYES Algorithm Accuracy Score : {:.2f}%".format(acc))


# DECISION TREE

# In[84]:


from sklearn.tree import DecisionTreeClassifier  
dt= DecisionTreeClassifier()  
dt.fit(X_train,Y_train)  

acc = dt.score(X_train, Y_train)*100
accuracies['DECISION TREE'] = acc
print("DECISION TREE Algorithm Accuracy Score : {:.2f}%".format(acc))



# COMPARE MODELS

# In[85]:


colors = ["red","green","orange","yellow","blue","black"]
sns.set_style("whitegrid")

plt.figure(figsize=(16,5))
plt.title('COMPARING MODELS of Training data')
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# In[86]:


acc = lr.score(X_test, Y_split)*100
accuracies['LOGISTIC REGRESSION'] = acc
print("LOGISTIC REGRESSION Accuracy Score : {:.2f}%".format(acc))

acc = dt.score(X_test, Y_split)*100
accuracies['DECISION TREE'] = acc
print("DECISION TREE Accuracy Score : {:.2f}%".format(acc))

acc = nb.score(X_test, Y_split)*100
accuracies['NAIVE BAYES'] = acc
print("NAIVE BAYES Accuracy Score : {:.2f}%".format(acc))

acc = knn.score(X_test, Y_split)*100
accuracies['KNN'] = acc
print("KNN Accuracy Score : {:.2f}%".format(acc))

acc = svm.score(X_test, Y_split)*100
accuracies['SVM'] = acc
print("SVM Accuracy Score : {:.2f}%".format(acc))

acc = rf.score(X_test, Y_split)*100
accuracies['RANDOM FOREST'] = acc
print("RANDOM FOREST Accuracy Score : {:.2f}%".format(acc))


# In[87]:


colors = ["red","green","orange","yellow","blue","black"]
sns.set_style("whitegrid")

plt.figure(figsize=(16,5))
plt.title('COMPARING MODELS of TESTING data')
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()


# CONFUSION MATRIX
# 

# In[88]:


y_head_lr = lr.predict(X_test)
y_head_rf = rf.predict(X_test)
y_head_knn = knn.predict(X_test)
y_head_svm = svm.predict(X_test)
y_head_nb = nb.predict(X_test)
y_head_dtc = dt.predict(X_test)


# In[89]:


from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(Y_split,y_head_lr)
cm_knn = confusion_matrix(Y_split,y_head_knn)
cm_svm = confusion_matrix(Y_split,y_head_svm)
cm_nb = confusion_matrix(Y_split,y_head_nb)
cm_dtc = confusion_matrix(Y_split,y_head_dtc)
cm_rf = confusion_matrix(Y_split,y_head_rf)


# In[90]:


plt.figure(figsize=(15,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# WEB APP

# In[91]:


import pickle


# In[92]:


filename = 'trainedfinalheartmodel1.sav'
pickle.dump(rf, open(filename, 'wb'))


# In[93]:


loaded_model = pickle.load(open('trainedfinalheartmodel1.sav', 'rb'))


# In[94]:


input_data = (57,1,4,130,131,0,0,115,1,1.2,2)
input_data_as_numpy_array = np.asarray(input_data)
input_reshape = input_data_as_numpy_array.reshape(1,-1)


prediction = loaded_model.predict(input_reshape)
print(prediction)



if (prediction[0]==0):
  print('The person does not heart disease')
else:
  print('The person has heart disease')


# In[ ]:





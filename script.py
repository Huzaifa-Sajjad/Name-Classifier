import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics

#1 Read the dataset into a dataframe
df = pd.read_csv("./baby_names.csv")

#2 Split the dataset into testing and training sets
X = df["NAME"]
y = df["GENDER"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#3 Creating a piple line to parse X Features and to train the model
name_clf = Pipeline([("tfidf",TfidfVectorizer()),("svc",LinearSVC())])
name_clf.fit(X_train,y_train)
 
 #4 Evaluation
predictions = name_clf.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

#Deployment
name = "christiano"
print(name_clf.predict([name]))
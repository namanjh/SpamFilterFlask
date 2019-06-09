import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
	

df = pd.read_csv("dataset.csv")
data = df[["CONTENT","CLASS"]]

'''data preprocessing'''
df_X = data.iloc[:,0].values
df_X = df_X.astype('str')
y = data.iloc[:,1].values

'''cleaning the text before using countvectorizer'''
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(len(df_X)):
    #comment = re.sub('http\S+',' ',df_X[i])
    comment = re.sub('[^a-zA-Z]',' ', df_X[i])
    comment = comment.lower()
    comment = comment.split(' ')
    ps = PorterStemmer()
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = " ".join(comment)
    corpus.append(comment)

'''extracting the features'''
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

'''splitting the dataset'''
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

'''building the predictive model'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

'''predicting the set of results'''
y_pred = classifier.predict(X_test)
'''the accuracy score of the model'''
classifier.score(X_test,y_test)

'''creating the confustion matrix for accuracy check'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print("Confusion Matrix: ",cm)

'''to save the model for using in predict module'''
from sklearn.externals import joblib
import pickle
joblib.dump(classifier,"saved_model.pkl")
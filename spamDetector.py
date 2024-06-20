import numpy as np
import pandas as pd
import matplotlib as plt
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score

import pickle

#download stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')

#import data from combinedSpamDataSet
df = pd.read_csv('SpamDataSet.csv',sep=";",encoding="ISO-8859-1")

#data preprocessing 
df['Message'] = df['Message'].apply(lambda x: x.replace("\r\n"," ")) #removing new line regex
df['Message'] = df['Message'].apply(lambda x: x.replace('\n'," ")) #removing new line regex

#remove duplicate
df = df.drop_duplicates(keep='first')
#setting up Lemmatizer
Lemmatizer = WordNetLemmatizer()

#set up stopwords
stop_words = set(stopwords.words('english'))

#corpus holder for cleaned text/message
corpus = []

#transvers through data set and do data cleaning
for i in range(len(df)):
    Message = df['Message'].iloc[i].lower() #make every word to lower case
    Message = Message.translate(str.maketrans('','',string.punctuation)).split() #removing punction 
    Message = [Lemmatizer.lemmatize(word) for word in Message if word not in stop_words] #removing stop words
    Message = ' '.join(Message)
    corpus.append(Message)

df['Message'] = corpus
df = df.sample(frac=1, random_state=42)#randomize dataset

#create a train/test set from original data
X = df['Message']
y = df['label']
x_train ,x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#vectorize Message data into numeric representation
TF_IDF = TfidfVectorizer()
x_train_count = TF_IDF.fit_transform(x_train.values)

#Train naive bayes model
model = MultinomialNB()
model.fit(x_train_count,y_train)

#test model
x_test_count = TF_IDF.transform(x_test)
print(f"Accuracy = {model.score(x_test_count,y_test)}")
y_predict = model.predict(x_test_count)
print(f"precision score =  {precision_score(y_test,y_predict)}")

#dump model data 
pickle.dump(TF_IDF,open("vectortizer.pkl","wb"))
pickle.dump(model,open("model.pkl","wb"))


#spam test
spam = "Urgent Prize Claim! Dear Winner, Congratulations! You've won a brand-new luxury car. To claim your prize, click the link below: Claim Your Prize Now: Click Here Don't miss out on this incredible opportunity! Act fast! Best regards, The Prize Team"
spam = spam.translate(str.maketrans('','',string.punctuation)).split()
spam = [Lemmatizer.lemmatize(word) for word in spam if word not in stop_words] 
spam = ' '.join(spam)
print(spam)
spam_test = []
spam_test.append(spam.lower())
spam_count = TF_IDF.transform(spam_test)
print(model.predict(spam_count))

#spam test
spam = "Congratulations! You’ve won a $1,000 Amazon gift card. Click now to claim: Claim Your Prize"
spam = spam.translate(str.maketrans('','',string.punctuation)).split()
spam = [Lemmatizer.lemmatize(word) for word in spam if word not in stop_words] 
spam = ' '.join(spam)
print(spam)
spam_test = []
spam_test.append(spam.lower())
spam_count = TF_IDF.transform(spam_test)
print(model.predict(spam_count))

#ham test
ham = "Hi John, just wanted to remind you about our meeting tomorrow at 10 AM. See you then!"
ham = ham.translate(str.maketrans('','',string.punctuation)).split()
ham = [Lemmatizer.lemmatize(word) for word in ham if word not in stop_words] 
ham = ' '.join(ham)
print(ham)
ham_test = []
ham_test.append(ham.lower())
ham_count = TF_IDF.transform(ham_test)
print(model.predict(ham_count))

#ham test
ham = "Hi Sarah, just wanted to let you know that the project deadline has been extended by a week. Take your time and ensure the quality of your work. Thanks!"
ham = ham.translate(str.maketrans('','',string.punctuation)).split()
ham = [Lemmatizer.lemmatize(word) for word in ham if word not in stop_words] 
ham = ' '.join(ham)
print(ham)
ham_test = []
ham_test.append(ham.lower())
ham_count = TF_IDF.transform(ham_test)
print(model.predict(ham_count))

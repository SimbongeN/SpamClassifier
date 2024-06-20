"""
deploying spam model as web application
"""
import streamlit as st
import pandas as pd
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#download stopwrds and wordnet
nltk.download('stopwords')
nltk.download('wordnet')

#load vectorizer and model
vectorizer = pickle.load(open("vectortizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

#header
st.header("Spam Detector Appliction :sunglasses:",divider='rainbow')

#greeting message
st.write(f'''
         Welcome to our Spam Classifier! 📧🔍 
         This application helps you determine whether an email , sms or message 
         is likely to be spam or not. Simply input the email, sms content or content of the message, and our model 
         will analyze it for you. Let’s keep those inboxes clean!
''')

#developer details
st.write("Developed by :rainbow[Simbonge Ndlovu]")
st.link_button("Visit My GitHub", "https://github.com/SimbongeN")

st.divider()#content divider

# display statictics of data such as how
# much data was the model trained with acc and precision
modelData = pd.read_csv('SpamDataSet.csv',sep=";",encoding="ISO-8859-1")
num_rows = modelData.shape[0] #get number of data used
category = modelData['Category'].value_counts()
spam_count = category.get('spam')
orginal = spam_count

#Display the Models data
col1, col2, col3= st.columns(3)
col1.metric("Training Data", str(num_rows)+"+")
col2.metric("Model Accuracy", "95.9%")
col3.metric("Model Precision", "98.3%")

st.divider()#content divider 

#make user test model by enetering email, sms or message content
st.subheader("Classify Content")
user_input = st.text_area("Message to analys",placeholder="Enter your Email, sms or Message content here",height=250)#user textarea

#preprocess the data given by the user
cleanedMessage = ''.join(map(lambda x: x.replace("\r\n", " "), user_input)) #removing new line regex
cleanedMessage = ''.join(map(lambda x: x.replace("\r\n", " "), cleanedMessage)) #removing new line regex

Lemmatizer = WordNetLemmatizer() #setting up Lemmatizer
stop_words = set(stopwords.words('english'))#set up stopwords

#remove all unwanted punctions in data
Message = cleanedMessage.lower() #make every word to lower case
Message = Message.translate(str.maketrans('','',string.punctuation)).split() #removing punction 
Message = [Lemmatizer.lemmatize(word) for word in Message if word not in stop_words] #removing stop words
Message = ' '.join(Message)

#vectorize user input
vector_input = vectorizer.transform([Message])
#model prediction classify user data
result = model.predict(vector_input)

#display user data
diff = 0
if st.button(":red[Classify ]"):
    if result == 1:
        st.subheader("SPAM")
        spam_count += 1
        diff = orginal - spam_count
    else:
        st.subheader("NOT SPAM")

st.divider()#content divider

st.metric("Spam Detected", spam_count, str(diff)+"+")
    

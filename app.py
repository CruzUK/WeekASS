import pandas as pd 
import streamlit as st
#from tensorflow import keras
from PIL import Image
#from keras import pad_sequences
import keras
from keras.utils import pad_sequences
import pickle 

#load ann model (line under mark it out if you using this one pick one. )
model=keras.models.load_model(r'C:\Users\Maciej JAROS\.vscode\WorkAss\ann_model')
#load the logistic regression model ann model or this choose

#with open(r'C:\Users\Maciej JAROS\.vscode\WorkAss\logistic_regression_model.pk1', 'rb') as file:
    #model_LR = pickle.load(file)

## load a copy of the data set

df=pd.read_csv('emails.csv')

## set the page configuration

st.set_page_config(page_title='Email Classifier', layout='wide')

##add page title and contents

st.title('Email Classifier using Artificial Neural Network')
st.write('Please  enter an email to be clasiffied: ')

## add image

#image = Image.open('datascience.jpg')
image=Image.open(r'C:\Users\Maciej JAROS\.vscode\WorkAss\datascience.jpg')
st.image(image, use_column_width=True ,caption='Data Science')

## user input

email_txt = st.text_input('Email text:')

## convert text to numerical values

word_index={word: index for index , word in enumerate(df.columns[:-1])}

numerical_email = [word_index[word] for word in email_txt.lower().split() if word in word_index]

## pad the numerical emails that it can have unique shape as the training data 

padded_email =pad_sequences([numerical_email], maxlen=3000)

#make prediction

##print the result

## set the threshold of 0.5
if st.button('Predict'):
    prediction=model.predict(padded_email)

    if prediction > 0.5:
        st.write('This email is spam')
    else:
        st.write('This email is not spam')









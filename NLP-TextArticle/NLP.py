# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:03:46 2022

@author: Tuf
"""
import os
import json
import re
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from module_for_NLP import ModelCreation,model_evaluation
#%% Statics
CSV_URL='https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'
TOKENIZER_PATH=os.path.join(os.getcwd(),'Model','tokenizer_NLP.json')
OHE_PATH =os.path.join(os.getcwd(),'Model','ohe.pkl')
LOG_PATH=os.path.join(os.getcwd(),'Model','Logs',datetime.datetime.now().\
                      strftime("%Y%m%d-%H%M%S"))
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'Model','model.h5')
#%% EDA

#Step 1: Data loading
df=pd.read_csv(CSV_URL)
#%% Step 2: Data inspection

#1) Statistical summary 
df.head(10)
df.tail(10)
df.info()
df.describe()

df['category'].unique() # to get the unique target.
df['text'][5] # to slice the first row in the review column
df['category'][5]

#2) Duplicates
df.duplicated().sum()
df[df.duplicated()]

#3) NaNs
df.isna().sum()

#4) Inspect numbers and characters
#%% Step 3: Data Cleaning

#1) Remove duplicates
df= df.drop_duplicates()
df.duplicated().sum()

#2) Remove numbers and unknown  characters
text=df['text'].values #feature: X
category=df['category'].values #target:y


for index,texts in enumerate(text):
    text[index]=re.sub('<.*?>',' ',texts) #(filter,replace with, data)
    text[index]=re.sub('[^a-zA-Z]',' ',texts).lower().split()
    
#%% Step 4: Features Selection

# For this dataset, there is nothing to be selected. 
# The target variable is category while, the feature variable is only text.

#%% Step 5: Pre-processing

#1) Convert into lower case
#2)Tokenizer
vocab_size=2000
oov_token='OOV'

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)

#tokenized the train data
tokenizer.fit_on_texts(text) 

# To get the training data word index
word_index=tokenizer.word_index
print(word_index) #le.fit

#Encode training data sentences into sequences
train_sequences= tokenizer.texts_to_sequences(text) # convert text to int

#To show the converted texts
for index,sentences in enumerate(train_sequences):
    print(text[index])
    print(sentences)

token_json=tokenizer.to_json()
with open(TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)
    
#3)Padding & Truncating
len(train_sequences[20]) 
# to get the sizes of each rows, each rows have different sizes.
# Thus, need to do padding to standardize the length of word in all rows
# To do that, use median/average to find the best no for padding

length_of_text=[len(i) for i in train_sequences]
np.median(length_of_text)
np.mean(length_of_text) # to get the no of max length for padding
#333.0

max_len=333
padded_text=pad_sequences(train_sequences,maxlen=max_len,padding='post',
                            truncating='post')

#4) OHE for target(category)
ohe=OneHotEncoder(sparse=False)
category=ohe.fit_transform(np.expand_dims(category,axis=-1))

with open(OHE_PATH,'wb') as file:
    pickle.dump(ohe,file) 

#5)Train Test Split
X_train,X_test,y_train,y_test= train_test_split(padded_text,category,
                                                test_size=0.3,
                                                random_state=123)
# To make sure that the X_train is in 3d shape
X_train=np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development

nb_classes=len(np.unique(y_train,axis=0)) #5
nb_features = np.shape(X_train)[1:] #(333,1)

mc=ModelCreation()
model=mc.Sequential_layer(input_node=333,vocab_size=2000,num_node=64,
                      drop_rate=0.2,output_node=5, embedding_dim=100)

plot_model(model,show_layer_names=(True), show_shapes=(True))

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics='acc')

# callbacks
#1) tensorboard callback
TensorBoard_callback = TensorBoard(log_dir=LOG_PATH)

#2) earlystopping
early_stopping_callback = EarlyStopping(monitor='loss',patience=3)

#%% Model training
hist = model.fit(X_train,y_train,
                  batch_size=64,
                  epochs=100,
                  validation_data=(X_test,y_test),
                  callbacks=[TensorBoard_callback,early_stopping_callback])

#%% Model Evaluation
hist.history.keys()

me=model_evaluation()
me.plot_graph(hist)

#%% model evaluationn

y_true=y_test
y_pred=model.predict(X_test)

y_true=np.argmax(y_true,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

#%% Model saving
model.save(MODEL_SAVE_PATH)
   
#%% Discussion

# The model achieved 92% in accuracy in categorizing the unseen articles into 
# 5 categories namely Sport, Tech, Business, Entertainment and Politics.   
# Despite the additional of nodes, LSTM and Bidirectional layers in the model 
# creation, the model's performance improved significantly but the learning 
# curve was overfit after multiple trainings with adjustments on the model 
# creation as visualized in the Tensorboard and Spyder.   
# The learning curve was overfitted as indicated by a large distanct between
# the training and validation losses without flattening as the training increases.
# As for the accuracy,training accuracy is slightly higher than validation accuracy, 
# which is also typical to an overfit model.
# To overcome the overfitting learning model, adding additional training data
# may help.
# Despite slightly overfitted, the model can be considered a good predictive 
# model as indicated by the f1 score with a value of 0.92 which is also
# proven in the deployment of the model.
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:36:44 2022

@author: Tuf
"""

import re
import os
import json
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

#%% Statics
TOKENIZER_PATH=os.path.join(os.getcwd(),'Model','tokenizer_NLP.json')
loaded_model=load_model(os.path.join(os.getcwd(),'Model','model.h5'))
OHE_PATH= os.path.join(os.getcwd(),'Model','ohe.pkl')
#%% Loading of Trained model, tokenizer, OHE
#1) To load trained model
loaded_model.summary()

#2) to load the tokenizer
with open(TOKENIZER_PATH,'r') as json_file:
    loaded_tokenizer=json.load(json_file) 

#3) to load ohe
with open(OHE_PATH,'rb') as file:
    loaded_ohe=pickle.load(file)
    
#%% Deployment

input_text=input('Please insert the text article :')

#preprocessing
input_text=re.sub('<.*?>',' ',input_text) #(filter,replace with, data)
input_text=re.sub('[^a-zA-Z]',' ',input_text).lower().split()

tokenizer=tokenizer_from_json(loaded_tokenizer)
input_text_encoded=tokenizer.texts_to_sequences(input_text)

input_review_encoded=pad_sequences(np.array(input_text_encoded).T,maxlen=333,
                                   padding='post',truncating='post')

#prediction
outcome=loaded_model.predict(np.expand_dims(input_text_encoded,axis=-1))
# np.argmax(outcome)
# ohe=OneHotEncoder()
# ohe.inverse_transform(outcome)

print("The article text is categorized into " + loaded_ohe.inverse_transform(outcome))

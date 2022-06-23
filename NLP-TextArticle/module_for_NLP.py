# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:54:09 2022

@author: Tuf
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional, Embedding
import matplotlib.pyplot as plt

class ModelCreation():
    def __init__(self):
        pass
    
    def Sequential_layer(self,input_node=333,vocab_size=2000,num_node=64,
                          drop_rate=0.2,output_node=5, embedding_dim=100):
        model = Sequential()
        model.add(Input(shape=(input_node)))
        model.add(Embedding(vocab_size, embedding_dim))
        model.add(Bidirectional(LSTM(num_node)))
        model.add(Dropout(drop_rate))
        model.add(Dense(num_node,activation='relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(output_node,'softmax'))
        model.summary()
        
        return model

class model_evaluation:
        def plot_graph(self,hist):
            plt.figure()
            plt.plot(hist.history['loss'],'r--',label='training loss')
            plt.plot(hist.history['val_loss'],label='Validation loss')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(hist.history['acc'],'r--',label='training acc')
            plt.plot(hist.history['val_acc'],label='Validation acc')
            plt.legend()
            plt.show()
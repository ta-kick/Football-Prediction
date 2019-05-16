import os
import pandas as pd
from keras.layers import Input,Dense, LSTM,GRU,Activation
from keras.layers.core import Dense,Dropout
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(file):
    dropcol=['wdl','vslank2','vslank3']
    droprow=['teamid','hora','esplank','vslank1','vslank2','vslank3']

    dataset=pd.read_csv(file, sep=",",names=["teamid","hora","wdl","esplank","vslank1","vslank2","vslank3"])
    X=dataset[(dataset["wdl"]!=1) & (dataset["esplank"]!=0)].drop(dropcol,axis=1).values
    Y=dataset[(dataset["wdl"]!=1) & (dataset["esplank"]!=0)].drop(droprow,axis=1).values.reshape(-1,1)
    
    idmap={"malaga":3,"valencia":1,"barcelona":1,"oviedo":3,
           "sevilla":1,"atletico":1,"rayo":3,"sociedad":2,
           "celta":3,"zaragoza":3,"racing":3,"mallorca":3,
           "alaves":2,"betis":2,"valladolid":3,"madrid":1,
           "numancia":4,"athletic":2,"deportivo":3,
           "osasuna":3,"palmas":3,"villarreal":2,
           "tenerife":4,"recreativo":3,"albacete":4,
           "murcia":3,"getafe":2,"levante":2,"cadiz":4,
           "tarragona":4,"almeria":4,"sporting":3,"xerez":4,
          "hercules":4,"granada":4,"elche":4,"eibar":2,
           "cordoba":4,"leganes":3,"girona":3}

    for idx in X:
        for idm in idmap:
            if idx[0]==idm:
                idx[0]=idmap[idm]
                
    return X,Y

def ffnn_model(HIDDEN_SIZE,input_size):
    model=Sequential()
    model.add(Dense(HIDDEN_SIZE,input_dim=input_size))
    model.add(Activation("relu"))
    model.add(Dense(1,input_dim=HIDDEN_SIZE))
    model.add(Activation("sigmoid"))
    return model

def lstm_model(HIDDEN_SIZE,NUM_STEPS):
    model=Sequential()
    model.add(LSTM(HIDDEN_SIZE, input_shape=(NUM_STEPS,1)))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    return model

def gru_model(HIDDEN_SIZE,NUM_STEPS):
    model=Sequential()
    model.add(GRU(HIDDEN_SIZE,input_shape=(NUM_STEPS,1)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def main(model):
    name=model
    file_path="data/espanyol.csv"
    X,Y=load_data(file_path)
    minmax=MinMaxScaler(copy=False)
    X=minmax.fit_transform(X)
    Y=minmax.fit_transform(Y)
    
    #train
    if model=="LSTM" or model=="GRU":
        X=np.expand_dims(X,axis=2)

    if model=="FFNN":
        NUM_EPOCHS=50
        HIDDEN_SIZE=70
        BATCH_SIZE=10
        input_size=4
        model=ffnn_model(HIDDEN_SIZE,input_size)
        
    elif model=="LSTM":
        NUM_STEPS=4
        HIDDEN_SIZE=90
        BATCH_SIZE=10
        NUM_EPOCHS=100
        model=lstm_model(HIDDEN_SIZE,NUM_STEPS)
        
    elif model=="GRU":
        NUM_STEPS=4
        HIDDEN_SIZE=90
        BATCH_SIZE=10
        NUM_EPOCHS=100
        model=gru_model(HIDDEN_SIZE,NUM_STEPS)
    
    xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2)
    model.compile(loss="binary_crossentropy",optimizer="SGD",metrics=['accuracy'])
    model.fit(xtrain,ytrain,epochs=NUM_EPOCHS,batch_size=BATCH_SIZE,validation_data=(xtest,ytest))
    
    #predict
    realdata=np.array([[2,1,8,9],[2,0,4,18],[2,1,7,5],[1,0,4,2],[2,1,9,12],[2,1,7,12],[3,0,5,20],[2,1,5,16],[1,0,2,3],
                       [3,1,5,9],[2,0,5,12],[1,1,8,1],[2,1,10,7],[1,0,11,3],[3,1,14,16],[2,0,8,11],[2,0,10,16],[1,1,13,3],
                       [3,1,15,18],[3,1,14,16],[1,1,11,6],[1,0,13,1],[3,0,13,14],[2,1,13,7],[1,1,10,2],[3,0,9,12]])
    
    future=np.array([[2,1,9,8]])
    
    
    realdata=minmax.fit_transform(realdata)
    future=minmax.fit_transform(future)
    
    if name=="LSTM" or name=="GRU":
        realdata=np.expand_dims(realdata,axis=2)
        future=np.expand_dims(future,axis=2)
    realpred=model.predict(realdata)
    future=model.predict(future)
    realresult=[["Valencia",1,"02"],["Alaves",0,"03"],["Levante",1,"04"],["RealMadrid",0,"05"],["Eibar",1,"06"],["Villarreal",1,"08"],
                ["Huesca",1,"09"],["Athletic club",1,"11"],["Sevilla",0,"12"],["Girona",0,"13"],["Getafe",0,"14"],["Barcelona",0,"15"],
                ["Betis",0,"16"],["Atletico Madrid",0,"17"],["Leganes",1,"18"],["Real sociedad",0,"19"],["Eibar",0,"20"],["Real Madrid",0,"21"],
                ["Rayo Vallecano",1,"23"],["Real Vallarid",1,"26"],["Sevilla",0,"28"],["Barcelona",0,"29"],["Girona",1,"31"],["Alaves",1,"32"],
                ["Atletico Madrid",1,"36"],["Leganes",1,"37"]]
    f_result=["Real sociedad","?",38]
               
    
    scount=count=0
    pred=[]
    
    for i in realpred:
        if i>=0.50:
            pred.append(1)
        else:
            pred.append(0)

    print("La Liga")
    print("Espanyol Predictions (*exclude tie games.)\n")
    
    print("Espanyol vs 2018-2019season\n")
    
    for h,i in zip(pred,realresult):
        if h==i[1]:
            scount+=1
            acc="〇"
        else:
            acc="×"
        count+=1
        
        if h==0:
            p="Lose"
            if i[1]==0:
                r="Lose"
            else:
                r="Win  "
        else:
            p="Win"
            if i[1]==0:
                r="Lose"
            else:
                r="Win  "   
        
        print("Week",i[2],i[0]," "*(16-len(i[0])),"predict:",p," ","real:",r,acc)
    
    if future[0]>=0.5:
        p="Win "
    else:
        p="Lose"
    print("Week",f_result[2],f_result[0]," "*(16-len(f_result[0])),"predict:",p," ","real:","?   ","?")
        
    accuracy=scount*100/count
    
    print("\nmodel: ",name)
    print("Accuracy=",'{0:.2f}'.format(scount*100/count),"%")

    
    
if __name__=="__main__":
    #select FFNN or LSTM or GRU.
    main("GRU") 
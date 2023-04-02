import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D , Flatten , Dropout
from keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import os

from matplotlib import pyplot as plt

def Cov_model():
    model = Sequential()
    model.add(Conv1D(32, 2, padding='valid', input_shape=(10, 9), activation="selu"))
    model.add(Conv1D(64, 2, padding='valid', activation="selu"))
    model.add(Conv1D(128, 2, padding='valid', activation="selu"))
    model.add(Conv1D(256, 2, padding='valid', activation="selu"))
    #model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    opt = Adam(lr=0.001, clipnorm=1)
    model.compile(optimizer=opt, loss=tf.keras.losses.mean_squared_error,
                  metrics=['mae'])
    model.summary()
    return model

# 加载网络
cov_model_put = Cov_model()
cov_model_call = Cov_model()

cov_model_put.load_weights('data/network/cov-mode-2023-2-22-13-09-02.h5')
cov_model_call.load_weights('data/network/cov-mode-2023-2-22-14-30-16.h5')

#拼预测数据
csv_folder = r'.\data\out'
output_folder = r'.\data'
csv_list = os.listdir(csv_folder)
output_path = output_folder + "\\" + 'final_data.csv'

call_X = []
call_Y = []
put_X = []
put_Y = []
# 加一个混合
# 做一个对比

#numbers = 1139

for csv_name in csv_list:
    if int(csv_name.split('.')[0]) % 100 == 0:
        print(csv_name)
    perdata = []
    '''numbers -= 1
    if numbers == 0:
        break'''
    csv_path = csv_folder +"\\"+csv_name
    data = pd.read_csv(csv_path, encoding = 'GBK', index_col= False)
    if len(data) < 11:
        continue

    datatype = data['call_or_put'][0]
    ll = 0

    for i in range(len(data)-1):
        onedata = list(data.iloc[i,3:])
        perdata.append(onedata)
        ll += 1
        if ll == 10:
            if datatype == 'call':
                call_X.append(perdata)
                call_Y.append(data['iv'][i+1])
                result = cov_model_call.predict(perdata)
            else:
                put_X.append(perdata)
                put_Y.append(data['iv'][i+1])
                result = cov_model_put.predict(perdata)
            ll -= 1
            perdata = perdata[1:]
            if result != 0:






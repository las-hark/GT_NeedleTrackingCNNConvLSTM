import os,keras
from matplotlib import pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D,BatchNormalization
from keras.layers import ConvLSTM2D
from keras.layers import TimeDistributed,Flatten
from keras.regularizers import l2


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x_train = np.load(r"E:\zyy\x_trains.npy")
y_train = np.load(r'E:\zyy\y_trains.npy')
x_val   = np.load(r'E:\zyy\x_valids.npy')
y_val   = np.load(r'E:\zyy\y_valids.npy')
x_test  = np.load(r'E:\zyy\x_testds.npy')
y_test  = np.load(r'E:\zyy\y_testds.npy')
# y_train = y_train/255
# y_test  = y_test/255
# y_val   = y_test/255
#模型参数-------------------------------------------------------------------------------------------
time_step = 5
# input_size = (3342,5,512,512,1)
# Convolution  卷积
k_size = (3,3)
# reg = l2(0.001)
model = Sequential()
model.add(TimeDistributed(Convolution2D(filters     = 8,
                                        kernel_size = k_size,
                                        activation  = 'relu',
                                        padding     = 'same'),input_shape=(5,256,256,1)))

model.add(TimeDistributed(MaxPooling2D(2)))
model.add(TimeDistributed(Convolution2D(filters     = 8,
                                        kernel_size = k_size,
                                        activation  = 'relu',
                                        padding     = 'same')))

model.add(TimeDistributed(MaxPooling2D(2)))
model.add(TimeDistributed(Convolution2D(filters     = 16,
                                        kernel_size = k_size,
                                        activation  = 'relu',
                                        padding     = 'same')))

# model.add(TimeDistributed(MaxPooling2D(2)))
# model.add(TimeDistributed(Convolution2D(filters     = 16,
#                                         kernel_size = k_size,
#                                         activation  = 'relu',
#                                         padding     = 'same')))

model.add(TimeDistributed(MaxPooling2D(2)))
model.add(ConvLSTM2D(filters = 16,
                     kernel_size = k_size,
                     padding = 'same',
                     activation  = 'relu',
                     return_sequences = True))
model.add(TimeDistributed(MaxPooling2D(pool_size = 2)))
model.add(ConvLSTM2D(filters = 8,
                     kernel_size = k_size,
                     padding = 'same',
                     activation  = 'relu',
                     return_sequences = True))
model.add(TimeDistributed(MaxPooling2D(pool_size = 2)))
model.add(ConvLSTM2D(filters = 4,
                     kernel_size = k_size,
                     padding = 'same',
                     activation  = 'relu',
                     return_sequences = False))
model.add(MaxPooling2D(pool_size = 2))
model.add(Flatten())
model.add(Dense(2))
print(model.summary())



model.compile(loss='mean_squared_error', optimizer='adam')

#模型保存-------------------------------------------------------------------------------------------
checkpoint_path = r'E:\zyy\cnnlstm\bestmodel.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = keras.callbacks.ModelCheckpoint(filepath = checkpoint_path,
                                              monitor='val_loss',
                                              verbose = 1,
                                              save_best_only=True,
                                              save_freq=1)
# if os.path.exists(checkpoint_path):
#     model.load_weights(checkpoint_path)
#     #若成功加载前面保存的参数，输出下列信息
#     print("checkpoint_loaded")

#模型训练-------------------------------------------------------------------------------------------
# Training   训练参数
batch_size = 16  # 批数据量大小
nb_epoch = 75    # 迭代次数

# 训练
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = nb_epoch,
          validation_data=(x_val, y_val), callbacks=[checkpoint])
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

# plot the training and validation loss
plt.plot(epochs, train_loss, '-', color = 'red',label='Training loss')
plt.plot(epochs, val_loss, '--', color ='green',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# save the final trained model
path_save = r'E:\zyy\cnnlstm\ori_model01.h5'
model.save(path_save)
np.save(r'E:\zyy\cnnlstm\trainloss_o1.npy',train_loss)
np.save(r'E:\zyy\cnnlstm\val_loss_o1.npy',val_loss)

model = keras.models.load_model(path_save)
PREDIC = model.predict(x_test)
ARR = np.absolute(PREDIC - y_test)
ARR_DIS=[]
for i in range(0,len(ARR)):
    ARR_DIS.append((ARR[i][0]**2 + ARR[i][1]**2)**0.5) 
mean_dis = np.mean(ARR_DIS)
STD_dis = np.std(ARR_DIS)
w = 0

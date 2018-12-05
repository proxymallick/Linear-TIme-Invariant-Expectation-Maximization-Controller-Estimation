
from __future__ import print_function
import numpy as np
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pickle

batch_size = 200
num_classes = 10
epochs = 130



def plot_trajectory(mean):
        
    x_em_test= mean.reshape(30,6)




    fig = plt.figure()

    ##########                 ###################                         ##################
    ## Plot the generated trajectory from the EM parameters                             ######
    ##########                 ###################                         ##################

    ax = fig.add_subplot(222, projection='3d')
    
    x =x_em_test[0:T,0].reshape(x_em_test.shape[0],)
    y =x_em_test[0:T,1].reshape(x_em_test.shape[0],)
    z =x_em_test[0:T,2].reshape(x_em_test.shape[0],)
        #ax.scatter(x, y, z, c='r', marker='o')
        #plt.pause(0.0001) 
    #ax.plot3D(x, y, z, 'gray')
    ax.scatter(x, y, z, c='b', marker='.')
    #plt.pause(0.0001)
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    plt.show()

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

T=1000
tot_data_size=2000

data_X=np.zeros((T,180))
data_X_init=np.zeros((T,180))

for i in range(T):
    #print i
    pkl_file_data = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/Sets of X_Em/X_EM%s.pkl'%(i+1), 'rb')
    pkl_file_data_init = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/Sets of X_initialised/X_init%s.pkl'%(i+1),'rb')

    #pkl_file_data = open('/home/prakash/gps/python/gps/dataset and pkl file/New_Test_GPS_EM_Combined/Sets of X_initialised/X_init%s.pkl'%(i+1),'rb')
    data_X_temp = (pickle.load(pkl_file_data)).reshape((180,))
    data_X_temp_init = (pickle.load(pkl_file_data_init)).reshape((180,))
    data_X[i,:]=data_X_temp
    data_X_init[i,:]=data_X_temp_init

######
##   Declare the total Input data set
######
X=np.zeros((T,6))
X_init=np.zeros((T,6))
#print data_X
for i in range(T):
    X[i,:]=data_X[i,:6]
    X_init[i,:]=data_X_init[i,:6]

X_total= np.vstack((X,X_init))

######
##   Declare the total Output data set
######
Y=data_X
Y_init=data_X_init
Y_total= np.vstack((Y,Y_init))


print (X_total.shape)
print (Y_total.shape)

#######
##   Declare training and testing dimensions
#######
train_num=1500#(.7*T).astype(int)#600
test_num=500#(.3*T).astype(int)
print (train_num)
print (test_num)



x_train = X_total[:train_num,:].reshape(train_num, 6)
x_test = X_total[train_num:tot_data_size,:].reshape(test_num, 6)


print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = Y_total[:train_num,:]

y_test = Y_total[train_num:tot_data_size,:]

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(6,)))
model.add(Dropout(0.2))
model.add(Dense(180, activation='relu'))
""" model.add(Dropout(0.2))
model.add(Dense(180, activation='softmax')) """

print (model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
X_predict = np.array([ -10.0,  20.0, 0.0, 0.0,  0.0, 0.00]).reshape(1,6)
mean = model.predict(X_predict)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
plot_trajectory(mean)



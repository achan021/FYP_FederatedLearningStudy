from tensorflow import keras
from tensorflow.python.keras import layers
import tensorflow as tf
import sklearn.metrics

def leNet_keras(c_out,c_in):
    model = keras.Sequential([
        layers.Conv2D(filters=6,kernel_size=5,input_shape=c_in,activation='relu'),
        layers.MaxPool2D(pool_size=(2,2),strides=2,padding='valid'),
        layers.Conv2D(filters=16,kernel_size=5,activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
        layers.Flatten(),
        layers.Dense(120,activation='relu'),
        layers.Dense(84,activation='relu'),
        layers.Dense(c_out,activation='softmax'),
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model

def dataloader_preprocess_keras():

    #download the train set
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
    #types of classes
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    x_train = x_train/255
    x_test = x_test/255

    return classes,x_train,y_train,x_test,y_test

def train_model_keras(net,x_train,y_train):
    # reset_keras()
    net.fit(x_train,
            y_train,
            epochs=2,
            batch_size=16)
    return net

def save_model(net):
    #save model weight
    save_path = '../database/keras_lenet_model.h5'
    net.save(save_path)
    #save model as json
    #serialize model to json
    net_json = net.to_json()
    save_path = '../database/keras_lenet_model.json'
    with open(save_path,'w') as json_file:
        json_file.write(net_json)


def load_model(PATH='../database/keras_lenet_model.h5'):
    net = keras.models.load_model(PATH)
    return net

def test_model(x_test,y_test,net):
    y_pred = net.predict_classes(x_test)
    print(sklearn.metrics.classification_report(y_test,y_pred))

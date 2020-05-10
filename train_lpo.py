#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import *
from multi_level_dam import Multi_DAM
from multi_level_dam import import_images
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from Get_Available_Gpus import get_available_gpus
import tensorflow as tf
import random

testAMD = [str(random.randint(1, 48)), str(random.randint(1, 48))]
testDME = [str(random.randint(1, 50)), str(random.randint(1, 50))]
testNormal = [str(random.randint(1, 50)), str(random.randint(1, 50))]

import_images = import_images()
train_image = import_images.train_images_from_folder('/home/bappaditya/Sapna/Rasti_data', \
                                                     testAMD, testDME, testNormal)
train_labels= np.array([i[1] for i in train_image])
train_labels_encoded=OneHotEncoder()
train_labels_one_hot=train_labels_encoded.fit_transform(train_labels.reshape(-1,1)).toarray()

tr_img_data=list([i[0] for i in train_image])
tr_img_data=np.array(tr_img_data)

test_image = import_images.test_images_from_folder('/home/bappaditya/Sapna/Rasti_data',\
                                                   testAMD, testDME, testNormal)
test_labels= np.array([i[1] for i in test_image])    
test_labels_encoded=OneHotEncoder()
test_labels_one_hot=test_labels_encoded.fit_transform(test_labels.reshape(-1,1)).toarray()

test_img_data=list([i[0] for i in test_image])
test_img_data=np.array(test_img_data)

X_train, X_val, y_train, y_val = train_test_split(tr_img_data, train_labels_one_hot, test_size=0.2)

batch_size=32   

train_datagen = ImageDataGenerator(
    width_shift_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=batch_size)


def get_callbacks(name_weights, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    return [mcp_save, reduce_lr_loss]

name_weights = "final_model_fold" + "_weights.h5"
callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

gpus = get_available_gpus(1)

with tf.device(gpus[1]):
    multi_dam = Multi_DAM(classes = 3, hidden_dim = 512)
    custom_dual_att_model =  multi_dam.model()
    opt =SGD(lr=0.001)    
    custom_dual_att_model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
    custom_dual_att_model.fit_generator(
        train_generator,
        steps_per_epoch = len(X_train)//batch_size,
        epochs=100,
        validation_data=(X_val,y_val), validation_steps=len(X_val)//batch_size, 
        callbacks=callbacks, shuffle=True)
    
    score=custom_dual_att_model.evaluate(test_img_data, test_labels_one_hot, batch_size=batch_size)
    print("Accuracy = " + format(score[1]*100, '.2f') + "%")
    y_predict=custom_dual_att_model.predict(test_img_data, batch_size=batch_size)
    y_pred=np.argmax(y_predict, axis=1)
    y_true=test_labels
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    print("Precision = ", precision*100+"%"),
    print("Recall = ", recall*100 +"%")    
    cc= confusion_matrix(y_true, y_pred)
    print("Confusion Matrix", cc)
    custom_dual_att_model.save('/home/bappaditya/Sapna/odes/multi_dam.hdf5')
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 20:36:07 2020

@author: sapna
"""
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input, Dropout, Lambda, Add, Reshape, \
                        AveragePooling2D, Average, Activation
from keras.engine.topology import Layer
import keras.backend as K
from keras_vggface.vggface import VGGFace
import numpy as np
import cv2
import os


nb_class = 3
hidden_dim = 512


class NormL(Layer):   
    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape
    

    
class Multi_DAM(object):  
    def __init__(self, classes, hidden_dim):
        self.nb_class = classes
        self.hidden_dim = hidden_dim
    def PAM(self, w=7, h=7, c=512, dout=512):   
        v1 = Input(shape = (w,h,c))
        q1 = Input(shape = (w,h,c))
        k1 = Input(shape = (w,h,c))
        
        v = Reshape([w*h,512])(v1)
        q = Reshape([w*h,512])(q1)
        k = Reshape([w*h,512])(k1)
            
        att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,2]),
                     output_shape=(w*h,w*h))([q,k]) # 49*49
        att = Lambda(lambda x:  K.softmax(x), output_shape=(w*h,w*h))(att)
    
        out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]),  \
                     output_shape=(w*h,512))([att,v])
        
        out = Reshape([w,h,512])(out)
        out = Add()([out, v1])   
        return  Model(inputs=[v1,q1,k1], outputs=out)
    
    def CAM(self, w=7, h=7, c=512):
        v1 = Input(shape = (w,h,c))
        q1 = Input(shape = (w,h,c))
        k1 = Input(shape = (w,h,c))
        
        v = Reshape([w*h,512])(v1)
        q = Reshape([w*h,512])(q1)
        k = Reshape([w*h,512])(k1)
        att= Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,1]), \
                     output_shape=(512,512))([q,k])
        out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2,1]),  \
                     output_shape=(w*h,512))([v,att])
        out = Reshape([w,h,512])(out)
        out = Add()([out, v1])
        return  Model(inputs=[v1,q1,k1], outputs=out)
    
    def model(self):
        vgg_model = VGGFace(include_top=False, input_shape=(112, 112, 3))
        x1 = vgg_model.get_layer('conv4_3').output
        x1 = AveragePooling2D()(x1)
        x2 = vgg_model.get_layer('conv4_1').output
        x2 = AveragePooling2D()(x2)
        x3 = vgg_model.get_layer('conv5_1').output
        xlast = vgg_model.get_layer('conv5_3').output
    
        if True:    
            att_1 = self.PAM()
            x_1 = att_1([x3,x2,x1])
            x_1 = Activation('relu')(x_1)
            x_1 = NormL()(x_1)
            att_2= self.CAM()
            x_2=att_2([xlast,xlast,xlast])
            x_2 = Activation('relu')(x_2)
            x_2 = NormL()(x_2)
        x=Average()([x_1,x_2])
        x=Flatten()(x)
        x = Dense(self.hidden_dim, activation='relu', name='fc6')(x)
        x=Dropout(0.25)(x)
        x = Dense(self.hidden_dim, activation='relu', name='fc7')(x)
        x=Dropout(0.25)(x)
        out = Dense(self.nb_class, activation='softmax', name='fc8')(x)
        model = Model(vgg_model.input, out)
        return model

class import_images(object):
    def import_images_from_folder(self, folder):
        images = []
        for index, name in enumerate(os.listdir(folder)):
            im_folder = os.path.join(folder, name)       
            for im in os.listdir(im_folder):
                img = cv2.imread(os.path.join(im_folder, im))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (512, 496))
                img = np.array(img).reshape(496, 512)
                img_1 = np.dstack((img, img, img))
                img = np.array(img_1).reshape(496, 512, 3)
                img = cv2.resize(img, (112, 112))
                img = np.array(img).reshape(112, 112, 3)

                if img is not None:
                    images.append((np.array(img), index))
        return images
    
    def train_images_from_folder(folder, testAMD, testDME, testNormal):
        images = []
        for index, name in enumerate(os.listdir(folder)):
            train_folder=os.path.join(folder, name)
            for ii, filename in enumerate(os.listdir(train_folder)):
                im_folder=os.path.join(train_folder, filename)
                if (name[0]== "A" and (filename[3:5] not in testAMD)) or \
                    (name[0]=="D" and (filename[3:5] not in testDME)) or \
                    (name[0]=="N" and (filename[3:5] not in testAMD)):   
                    for im in os.listdir(im_folder):
                        img=cv2.imread(os.path.join(im_folder,im))
                        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img=cv2.resize(img,(512,496))
                        img=np.array(img).reshape(496,512)
                        img_1=np.dstack((img,img,img))
                        img=np.array(img_1).reshape(496,512,3)
                        img= cv2.resize(img,(224,224))
                        img=np.array(img).reshape(224,224,3)
                        if img is not None:
                            images.append((np.array(img),index))
        return images
    
    def test_images_from_folder(folder, testAMD, testDME, testNormal):
        images = []
        for index, name in enumerate(os.listdir(folder)):
            train_folder=os.path.join(folder, name)
            for ii, filename in enumerate(os.listdir(train_folder)):
                im_folder=os.path.join(train_folder, filename)
                if (name[0]== "A" and (filename[3:5] in testAMD)) or \
                    (name[0]=="D" and (filename[3:5] in testDME)) or \
                    (name[0]=="N" and (filename[3:5] in testAMD)): 
                    for im in os.listdir(im_folder):
                        img=cv2.imread(os.path.join(im_folder,im))
                        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img=cv2.resize(img,(512,496))
                        img=np.array(img).reshape(496,512)
                        img_1=np.dstack((img,img,img))
                        img=np.array(img_1).reshape(496,512,3)    
                        img= cv2.resize(img,(224,224))
                        img=np.array(img).reshape(224,224,3)
                        
                        if img is not None:
                            images.append((np.array(img),index))                           
        return images
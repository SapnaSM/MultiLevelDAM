#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.optimizers import *
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.image import ImageDataGenerator
from scipy import interp
import tensorflow as tf
from multi_level_dam import Multi_DAM
from multi_level_dam import import_images
from Get_Available_Gpus import get_available_gpus
import numpy as np

gpus = get_available_gpus(1)
with tf.device(gpus[0]):
    DAM = Multi_DAM(classes = 4, hidden_dim = 512)
    custom_dual_att_model = DAM.model()
    opt =SGD(lr=0.001)    
    custom_dual_att_model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=['accuracy'])
    import_images = import_images()
    train_image = import_images.import_images_from_folder('/home/bappaditya/Sapna/Rasti_data/train')  
    tr_labels = [i[1] for i in train_image]
    tr_img_data_l = list([i[0] for i in train_image])

    test_image = import_images.import_images_from_folder('/home/bappaditya/Sapna/Rasti_data/val')
    test_labels = [i[1] for i in test_image]
    test_img_data_l = list([i[0] for i in test_image])

    train_img_data_l = tr_img_data_l + test_img_data_l
    train_img_data = np.array(train_img_data_l)

    min_max_scalar = MinMaxScaler()
    min_max_scalar.fit(train_img_data.reshape(len(train_img_data_l), 112 * 112 * 3))
    tr_img = min_max_scalar.transform(train_img_data.reshape(len(train_img_data_l),\
                                                             112 * 112 * 3))
    tr_img_data = tr_img.reshape(len(train_img_data_l), 112, 112, 3)
    train_labels = tr_labels + test_labels 
   
    batch_size = 32
    testgen = ImageDataGenerator()

    gen = ImageDataGenerator(
        width_shift_range=40,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    folds = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=32).\
                 split(tr_img_data, train_labels))

    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, \
                                   monitor='val_acc', mode='max')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, \
                                           patience = patience_lr, verbose=1, \
                                           epsilon=1e-4, mode='min')
        return [mcp_save, reduce_lr_loss]
    
    cvscores = []
    cvp = []
    cvr = []

    for j, (train_idx, val_idx) in enumerate(folds):
        X_train_cv = tr_img_data[train_idx]
        y_train_cv_1 = train_labels[train_idx]
        train_labels_encoded = OneHotEncoder()
        y_train_cv = train_labels_encoded.fit_transform(y_train_cv_1.\
                                                        reshape(-1, 1)).toarray()

        X_valid_cv = tr_img_data[val_idx]
        y_valid_cv_1 = train_labels[val_idx]
        valid_labels_encoded = OneHotEncoder()
        y_valid_cv = valid_labels_encoded.fit_transform(y_valid_cv_1.\
                                                        reshape(-1, 1)).toarray()

        name_weights = "final_model_oct17_1_fold" + str(j) + "_weights.h5"
        callbacks = get_callbacks(name_weights=name_weights, patience_lr=10)
        generator = gen.flow(X_valid_cv, y_valid_cv, batch_size=batch_size)
        custom_dual_att_model.fit_generator(
            generator,
            steps_per_epoch=len(X_valid_cv//batch_size),
            epochs=10,
            shuffle=True,
            verbose=2,
            validation_data=(X_train_cv, y_train_cv),
            validation_steps=len(X_train_cv//batch_size),
            callbacks=callbacks
        )

        score = custom_dual_att_model.evaluate(X_train_cv, y_train_cv, \
                                               batch_size=batch_size, verbose = 2)
        print("Accuracy = " + format(score[1] * 100, '.2f') + "%")
        y_predict = custom_dual_att_model.predict(X_train_cv, batch_size=batch_size)
        y_pred = np.argmax(y_predict, axis=1)
        y_true = y_train_cv_1
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        print("Precision = ", precision),
        print("Recall = ", recall)
        cc = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix", cc)
        cvscores.append(score[1] * 100)
        cvp.append(precision * 100)
        cvr.append(recall * 100)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(y_train_cv[:, i], y_predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        n_classes = 4
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        print("auc=", roc_auc["macro"])
        
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_dual_oct17_fpr.npy'.format(j), all_fpr)
        np.save('/home/bappaditya/Sapna/Codes/fold_{0}_dual_oct17_tpr.npy'.format(j), mean_tpr)
        custom_dual_att_model.save('/home/bappaditya/Sapna/Codes/fold_{0}_dual_oct17.hdf5'.format(j))

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvp), np.std(cvp)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvr), np.std(cvr)))
    
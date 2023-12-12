import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
from collections import Counter

from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                

def model_predict(model, hist, y_shuffle_train, y_shuffle_valid, y_shuffle_test, \
                  image_shuffle_test,text_shuffle_test,y_vrank_shuffle_test,y_lrank_shuffle_test):
    fig,ax=plt.subplots(1,2)

    ax[0].plot(hist.history['T1_accuracy'])
    ax[0].plot(hist.history['val_T1_accuracy'])
    ax[0].set_title('Popularity(binary) accuracy')
    ax[0].legend(['Train', 'Valid'], loc='upper left')

    ax[1].plot(hist.history['T1_loss'])
    ax[1].plot(hist.history['val_T1_loss'])
    ax[1].set_title('Popularity(binary) loss')
    ax[1].legend(['Train', 'Valid'], loc='upper left')
    plt.show()

    pred = model.predict([image_shuffle_test,text_shuffle_test,y_vrank_shuffle_test,y_lrank_shuffle_test])

    print("Train label 비율) non-popular:",Counter(y_shuffle_train)[0],", popular:",Counter(y_shuffle_train)[1])
    print("Valid label 비율) non-popular:",sum(y_shuffle_valid==0),", popular:",sum(y_shuffle_valid==1))
    print("Test label 비율) non-popular:",sum(y_shuffle_test==0),", popular:",sum(y_shuffle_test==1))
    print()

    pred_l = np.where(pred[0] > 0.5, 1 , 0)
    cm = confusion_matrix(y_shuffle_test, pred_l)
    tn, fp, fn, tp = cm.ravel()
    print("confusion matrix: "); print(cm)
    print(classification_report(y_shuffle_test, pred_l,\
                                target_names=['non-popular','popular'], digits=3))
    plot_confusion_matrix(cm)

    recall = tp/(tp+fn)
    specifi = tn/(tn+fp)
    print("G-Mean score: ", str(math.sqrt(recall*specifi)))
    print("Imbalanced metrics: ")
    print(classification_report_imbalanced(y_shuffle_test, pred_l,\
                                           target_names=['non-popular','popular'], digits=3))
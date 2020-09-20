#coding=utf-8
'''
Created on 2020-9-20

@author: Yoga
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix

mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss_both_for_train_val(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
                color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
                color=colors[n], label='Val '+label, linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
  
    plt.legend()
    
    
    
def plot_loss(history, label, loss, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history[loss],
                color=colors[n], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
  
    plt.legend()
    
    
    
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])#合法交易
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])#合法交易但被错认为是欺诈行为
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])#欺诈交易但被认为是合法的
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])#欺诈交易
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))#总的真实的欺诈交易数（y轴的总数）
    
    
    
    
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
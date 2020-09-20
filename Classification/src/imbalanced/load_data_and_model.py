#coding=utf-8
'''
Created on 2020-9-20

@author: Yoga
'''
import tensorflow as tf
from tensorflow import keras

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler





os.environ['CUDA_VISIBLE_DEVICES']='0'
#设置按需使用GPUs
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices(device_type='GPU')
        print('************************** ', len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)
        
        
        
#加载数据集   
data_dir = './data/'

raw_df = pd.read_csv(data_dir + 'creditcard.csv')
print(raw_df.head())

neg, pos = np.bincount(raw_df['Class'])#bincount(): Count number of occurrences of each value in array of non-negative ints
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))




#数据预处理，去除无意义数据，转化数据range
cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)





#划分数据集
# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

bool_train_labels = train_labels != 0

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)





#归一化训练特征（fit时仅使用训练集）
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)#为什么要clip？因为fit时只使用了train数据集，val和test肯定会有些差异，transform后不一定都落在一个0附近的区间
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)








#搭建模型
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None, loss_func=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(train_features.shape[-1],)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])
    
    if loss_func is None:
        model.compile(
            optimizer=keras.optimizers.Adam(lr=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics)

    return model






EPOCHS = 100
BATCH_SIZE = 2048#尽量保证每个batch都至少包含正样本

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)#在每个epoch结束时检查指标是否更好了，是就保存当前最好模型的权重，当patience用完或训练结束结束时，模型会以bestweights重新被赋值

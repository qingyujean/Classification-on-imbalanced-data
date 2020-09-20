#coding=utf-8
'''
Created on 2020-9-20

@author: Yoga
'''

import numpy as np
import pandas as pd
import tensorflow as tf

from imbalanced.load_data_and_model import  neg, pos, make_model, \
    train_features, train_labels, BATCH_SIZE, EPOCHS, early_stopping, \
    val_features, val_labels, test_features, test_labels, METRICS



##公式：L(pt) = -αt(1-pt)^γ log(pt)，pt=p and αt=α  when y=1 ,pt=1-p and αt=1-α when y=-1或者0 视情况而定
def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*'*20, 'alpha={}, gamma={}'.format(alpha, gamma))
    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_probs, tf.ones_like(y_probs))
        
        loss =  -alpha * tf.pow(1-positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
            (1-alpha) * tf.pow(1-negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt,  epsilon, 1.))

        return tf.reduce_sum(loss)
    return focal_loss_calc




alphas = np.arange(0.1, 0.41, 0.05)#[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
gammas = np.arange(1., 4.1, 0.5)#[1.0, 1.5, 2., 2.5, 3., 3.5, 4.]


initial_bias = np.log([pos/neg])
model = make_model(output_bias = initial_bias, loss_func='focal_loss')
initial_weights = model.get_weights()#bias=np.log([pos/neg])

all_results = []

for i in range(len(alphas)):
    for j in range(len(gammas)):
        
        model.set_weights(initial_weights)#重新初始化模型
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=focal_loss(alpha=alphas[i], gamma=gammas[j]),
            metrics=METRICS,
            run_eagerly=True)##############

        focalloss_history = model.fit(
            train_features,
            train_labels,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks = [early_stopping],
            validation_data=(val_features, val_labels)
        ) 
        
        #评估
        focal_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
        
        focal_metric_res = {'alpha': alphas[i], 'gamma': gammas[j]}
        
        for name, value in zip(model.metrics_names, focal_results):
            print(name, ': ', value)
            focal_metric_res[name] = value
        print()

        all_results.append(focal_metric_res)


res_df = pd.DataFrame(all_results)
res_df.to_csv('./files/alphas_and_gammas.csv', sep=',', index=False, encoding='UTF-8')


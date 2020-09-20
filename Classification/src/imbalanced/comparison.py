#coding=utf-8
'''
Created on 2020-9-20

@author: Yoga
'''

from imbalanced.draw_helper import plot_loss_both_for_train_val, plot_roc, plot_cm
from imbalanced.load_data_and_model import  neg, pos, make_model, \
    train_features, train_labels, BATCH_SIZE, EPOCHS, early_stopping, \
    val_features, val_labels, test_features, test_labels, METRICS, total, \
    bool_train_labels
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 



#方法1：使用正确的bias
initial_bias = np.log([pos/neg])
model = make_model(output_bias = initial_bias)


#因为要使用各种方案来训练模型然后进行比较，所以把模型的初始权重保存下来，方便后面各种训练对比
initial_weights = model.get_weights()#bias=np.log([pos/neg])



#先看看不使用先验bias的效果
model.set_weights(initial_weights)
model.layers[-1].bias.assign([0.0])###bias重新赋值为为0
zero_bias_history = model.fit(
                        train_features,
                        train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=20,
                        validation_data=(val_features, val_labels), 
                        verbose=0)


#再看看使用先验bias的效果
model.set_weights(initial_weights)#bias=np.log([pos/neg])
careful_bias_history = model.fit(
                        train_features,
                        train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=20,
                        validation_data=(val_features, val_labels), 
                        verbose=0)



   
plot_loss_both_for_train_val(zero_bias_history, "Zero Bias", 0)
plot_loss_both_for_train_val(careful_bias_history, "Careful Bias", 1)
plt.savefig('./imgs/bias_helped.png')
plt.show()
#上图可以看出，bias有助于改善模型训练，模型的初期几个epoch不用再学习bias的变化

#接下来就以加了先验bias的模型作为baseline，分别验证其他各种解决样本不均衡的方法
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)


baseline_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)
plt.savefig('./imgs/baseline_cm.png')
plt.show()








#方法2：类别加权
weight_for_0 = (1./neg)*total/2.
weight_for_1 = (1./pos)*total/2.

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))


model.set_weights(initial_weights)#bias=np.log([pos/neg])

weighted_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    # The class weights go here
    class_weight=class_weight) #################################

train_predictions_weighted = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, weighted_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)#绘制混淆矩阵
plt.savefig('./imgs/class_weights_cm.png')
plt.show()






#方法3：上采样
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]
print('positive examples num : {:.2f}'.format(len(pos_labels)))
print('negative examples num : {:.2f}'.format(len(neg_labels)))

#重采样实现2：使用tf.data API
BUFFER_SIZE = 100000

def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)

for features, label in pos_ds.take(1):
    print("Features:\n", features.numpy())
    print()
    print("Label: ", label.numpy())

#合并两个数据集对象，并传入数据集对象的占比，各占0.5，在这个参数用以resample
resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
    print(label.numpy().mean())#计算label=1的样本数占总样本数的比例  此时应该近似于0.5
    
    
#此时需要定义一个epoch需要执行多少个steps，这里假设一个epoch至少要看到每个负样本一次 所需要的的batch数
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)#ceil 计算大于等于该值的最小整数
print('resampled_steps_per_epoch:', resampled_steps_per_epoch)


#注意：由于过采样，总的样本数增大了，每个epoch的训练时间自然也会增长

model.set_weights(initial_weights)#bias=np.log([pos/neg])

# Reset the bias to zero, since this dataset is balanced.
output_layer = model.layers[-1] ############数据集已经平衡，所以bias要重新赋值为0
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) #预取2个batch到memory，提高gpu处理和数据pipeline的并行度，加速


resampled_history = model.fit(
    resampled_ds,
    epochs=EPOCHS,
    steps_per_epoch=resampled_steps_per_epoch,
    callbacks = [early_stopping],
    validation_data=val_ds)



#评估
train_predictions_resampled = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = model.predict(test_features, batch_size=BATCH_SIZE)

resampled_results = model.evaluate(test_features, test_labels,
                                             batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, resampled_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_resampled)#绘制混淆矩阵
plt.savefig('./imgs/resampled_cm.png')
plt.show()






#方法4：focal loss
#FL(pt) = -αt(1-pt)^γ log(pt)，pt=p and αt=α  when y=1 ,pt=1-p and αt=1-α when y=-1或者0 视情况而定
def focal_loss(alpha=0.5, gamma=1.5, epsilon=1e-6):
    print('*'*20, 'alpha={}, gamma={}'.format(alpha, gamma))
    def focal_loss_calc(y_true, y_probs):
        positive_pt = tf.where(tf.equal(y_true, 1), y_probs, tf.ones_like(y_probs))
        negative_pt = tf.where(tf.equal(y_true, 0), 1-y_probs, tf.ones_like(y_probs))
        
        loss =  -alpha * tf.pow(1-positive_pt, gamma) * tf.math.log(tf.clip_by_value(positive_pt, epsilon, 1.)) - \
            (1-alpha) * tf.pow(1-negative_pt, gamma) * tf.math.log(tf.clip_by_value(negative_pt,  epsilon, 1.))

        return tf.reduce_sum(loss)
    return focal_loss_calc


best_alpha = 0.3
best_gamma = 2.
model = make_model(loss_func='focal_loss')
model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
            loss=focal_loss(alpha=best_alpha, gamma=best_gamma),
            metrics=METRICS,
            run_eagerly=True)##############
model.set_weights(initial_weights)#bias=np.log([pos/neg])


focalloss_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks = [early_stopping],
    validation_data=(val_features, val_labels),
    ) 


#评估
train_predictions_focal = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_focal = model.predict(test_features, batch_size=BATCH_SIZE)

focal_results = model.evaluate(test_features, test_labels, batch_size=BATCH_SIZE, verbose=0)

for name, value in zip(model.metrics_names, focal_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_focal)#绘制混淆矩阵
plt.savefig('./imgs/focalloss_cm.png')
plt.show()


#对比ROC曲线
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])#绘制roc曲线
plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])#绘制roc曲线
plot_roc("Train Focal_Loss", train_labels, train_predictions_focal, color=colors[3])#绘制roc曲线
plt.legend(loc='lower right')
plt.savefig('./imgs/training_roc_comparison.png')
plt.show()


plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')
plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')
plot_roc("Test Focal_Loss", test_labels, test_predictions_focal, color=colors[3], linestyle='--')
plt.legend(loc='lower right')
plt.savefig('./imgs/testing_roc_comparison.png')
plt.show()

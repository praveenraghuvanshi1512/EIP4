# EIP4 Assignment 3 - Praveen Raghuvanshi

#### Final Base Model Accuracy : 81.88%

```
Accuracy on test data is: 81.88
```

### Updated Model

- Val Accuracy(26th Epoch): **82.13%**
- Total parameters: **93,103**
- No of Epochs: 50
- ipynb file : [Solution Iteration Tenth](EIP4_A3_Praveen_Raghuvanshi_82_13.ipynb)

```python
Accuracy on test data is: 81.38
```

##### Model 

```python
# Define the model
# Replace Conv2D with SeparableConv2D
# Add BN before activation
# Remove D1(393,728) and D2(131,328)
# Remove Dropout from last layer 
# Reduce dropout (0.25 --> 0.1)
# Increase Batch size (128 --> 256)
# Reduce Batch size (128 --> 64)
# Reduce Batch size (128 --> 64 --> 32)

model = Sequential()
model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(48, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(SeparableConv2D(96, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(96, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(SeparableConv2D(192, 3, 3, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(192, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
# model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
# model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

##### Logs

```
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  if sys.path[0] == '':
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:12: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=1562, epochs=50)`
  if sys.path[0] == '':
Epoch 1/50
1562/1562 [==============================] - 52s 34ms/step - loss: 1.5006 - acc: 0.4692 - val_loss: 1.1766 - val_acc: 0.5778
Epoch 2/50
1562/1562 [==============================] - 49s 32ms/step - loss: 1.0869 - acc: 0.6148 - val_loss: 1.3856 - val_acc: 0.5499
Epoch 3/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.9266 - acc: 0.6750 - val_loss: 1.1001 - val_acc: 0.6277
Epoch 4/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.8399 - acc: 0.7072 - val_loss: 1.1926 - val_acc: 0.5851
Epoch 5/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.7716 - acc: 0.7289 - val_loss: 0.8589 - val_acc: 0.7047
Epoch 6/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.7278 - acc: 0.7454 - val_loss: 0.7567 - val_acc: 0.7446
Epoch 7/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.6894 - acc: 0.7623 - val_loss: 0.8404 - val_acc: 0.7118
Epoch 8/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.6549 - acc: 0.7727 - val_loss: 0.6880 - val_acc: 0.7693
Epoch 9/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.6218 - acc: 0.7807 - val_loss: 0.6882 - val_acc: 0.7714
Epoch 10/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.6023 - acc: 0.7887 - val_loss: 0.8096 - val_acc: 0.7203
Epoch 11/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.5774 - acc: 0.7989 - val_loss: 0.7869 - val_acc: 0.7339
Epoch 12/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.5598 - acc: 0.8052 - val_loss: 0.7210 - val_acc: 0.7520
Epoch 13/50
1562/1562 [==============================] - 49s 31ms/step - loss: 0.5399 - acc: 0.8112 - val_loss: 0.6293 - val_acc: 0.7868
Epoch 14/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.5233 - acc: 0.8168 - val_loss: 0.8097 - val_acc: 0.7263
Epoch 15/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.5135 - acc: 0.8200 - val_loss: 0.5829 - val_acc: 0.8040
Epoch 16/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4926 - acc: 0.8273 - val_loss: 0.6099 - val_acc: 0.7943
Epoch 17/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4858 - acc: 0.8295 - val_loss: 0.6096 - val_acc: 0.7930
Epoch 18/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.4743 - acc: 0.8326 - val_loss: 0.6852 - val_acc: 0.7671
Epoch 19/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.4601 - acc: 0.8389 - val_loss: 0.6589 - val_acc: 0.7778
Epoch 20/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4530 - acc: 0.8421 - val_loss: 0.6656 - val_acc: 0.7749
Epoch 21/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4442 - acc: 0.8445 - val_loss: 0.5613 - val_acc: 0.8125
Epoch 22/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.4392 - acc: 0.8432 - val_loss: 0.6378 - val_acc: 0.7865
Epoch 23/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4276 - acc: 0.8491 - val_loss: 0.5833 - val_acc: 0.8034
Epoch 24/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.4157 - acc: 0.8528 - val_loss: 0.5836 - val_acc: 0.8018
Epoch 25/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4159 - acc: 0.8538 - val_loss: 0.5729 - val_acc: 0.8090
Epoch 26/50
1562/1562 [==============================] - 49s 31ms/step - loss: 0.4047 - acc: 0.8574 - val_loss: 0.5353 - val_acc: 0.8213
Epoch 27/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.4005 - acc: 0.8567 - val_loss: 0.5498 - val_acc: 0.8121
Epoch 28/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3876 - acc: 0.8608 - val_loss: 0.5691 - val_acc: 0.8067
Epoch 29/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3858 - acc: 0.8630 - val_loss: 0.5922 - val_acc: 0.7960
Epoch 30/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3789 - acc: 0.8633 - val_loss: 0.5880 - val_acc: 0.8004
Epoch 31/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.3716 - acc: 0.8675 - val_loss: 0.6330 - val_acc: 0.7875
Epoch 32/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3706 - acc: 0.8682 - val_loss: 0.5608 - val_acc: 0.8086
Epoch 33/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3648 - acc: 0.8713 - val_loss: 0.5517 - val_acc: 0.8123
Epoch 34/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3572 - acc: 0.8726 - val_loss: 0.5566 - val_acc: 0.8088
Epoch 35/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3590 - acc: 0.8717 - val_loss: 0.5604 - val_acc: 0.8093
Epoch 36/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3484 - acc: 0.8780 - val_loss: 0.5369 - val_acc: 0.8207
Epoch 37/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.3433 - acc: 0.8785 - val_loss: 0.5594 - val_acc: 0.8103
Epoch 38/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3407 - acc: 0.8798 - val_loss: 0.6368 - val_acc: 0.7856
Epoch 39/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3355 - acc: 0.8800 - val_loss: 0.5365 - val_acc: 0.8166
Epoch 40/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3328 - acc: 0.8811 - val_loss: 0.5847 - val_acc: 0.8037
Epoch 41/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3288 - acc: 0.8835 - val_loss: 0.5771 - val_acc: 0.8049
Epoch 42/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3207 - acc: 0.8870 - val_loss: 0.5695 - val_acc: 0.8079
Epoch 43/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.3241 - acc: 0.8847 - val_loss: 0.5756 - val_acc: 0.8045
Epoch 44/50
1562/1562 [==============================] - 49s 31ms/step - loss: 0.3164 - acc: 0.8878 - val_loss: 0.5384 - val_acc: 0.8185
Epoch 45/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3177 - acc: 0.8863 - val_loss: 0.5366 - val_acc: 0.8186
Epoch 46/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3111 - acc: 0.8889 - val_loss: 0.5579 - val_acc: 0.8152
Epoch 47/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3148 - acc: 0.8880 - val_loss: 0.5671 - val_acc: 0.8096
Epoch 48/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3075 - acc: 0.8911 - val_loss: 0.5634 - val_acc: 0.8145
Epoch 49/50
1562/1562 [==============================] - 50s 32ms/step - loss: 0.3062 - acc: 0.8916 - val_loss: 0.5525 - val_acc: 0.8164
Epoch 50/50
1562/1562 [==============================] - 49s 32ms/step - loss: 0.3035 - acc: 0.8910 - val_loss: 0.5662 - val_acc: 0.8138
Model took 2473.99 seconds to train

Accuracy on test data is: 81.38
```



#### Analysis

##### Pre-Analysis

- Dataset : Cifar10 Dataset

- Image size : 32 x32 x 3

- Train data: 50000, Test data: 10000

- Classes (10) : ['airplane','automobile','bird','cat','deer', 'dog','frog','horse','ship','truck'] 

- Base Model

  - Parameters: 1,172, 410

  - No of layers : 26

  - Layer pattern:  

    - 3 Batch x [Conv(3x3) -> ReLU -> Conv(3x3) -> ReLU -> MP(2,2) -> Dropout(0.25) ] 
    - Flatten
    - 2 x [Dense(512, 256) -> ReLU -> Dropout(0.5)]
    - Last [Dense(10) + Softmax]

  - Optimizer : Adam

  - Loss : Categorical Cross Entropy

  - Val Acc: 81.88%, Train Acc: 88.70%. Difference - 6.82

  - Val loss: 0.6007, Train loss: 0.3292

  - Batch Size: 128

  - No of epochs: 50

  - Time/Epoch : 7s(Google colab - GPU)

  - Total Time: 342.05s ~ 6.2 min

  - Model is overfitting

    - As per graph curve for Val and train loss is diverging after some epochs
    - Need to reduce the parameters as model is mugging up

    

| S.No | Iteration | # Parameters | Val Acc (Best)     | # layers | Time(min) | Model Changes                                                | Remark                                   |
| ---- | --------- | ------------ | ------------------ | -------- | --------- | ------------------------------------------------------------ | ---------------------------------------- |
| 1    | Base      | 1,172, 410   | 81.88 (50th Epoch) | 26       | 6.2       | No Change                                                    | N/A                                      |
| 2    | First     | 604,213      | 100                | 26       | 10        | Conv2D --> SeparableConv2D                                   | Parameters reduced, Acc is constant 1.00 |
| 3    | Second    | 609,973      | 82.18              | 32       | 12.8      | Batch normalization (SepConv -> BN -> ReLU -> SepConv -> BN -> ReLU -> MP) | Parameters increased, Accuracy is normal |
| 4    | Third     | 93,109       | 78.80              | 30       | 12.9      | Remove Dense Layer D1(393,728) and D2(131,328)               | Parameters is under 100,000, Acc reduced |
| 5    | Fourth    | --           | 80.50              | 29       | 12.9      | Remove Dropout from last layer                               | Acc improved                             |
| 6    | Fifth     | --           | 80.92              | 29       | 12.9      | Reduce Dropout from 0.25 to 0.1                              | Acc improved                             |
| 7    | Sixth     | --           | 78.98              | 27       | 12.9      | Remove all Dropout                                           | Acc reduced                              |
| 8    | Seventh   | --           | 79.91              | 29       | 20        | Increase Batch size (128 --> 256)                            | Acc improved                             |
| 9    | Eighth    | --           | 77.93              | 29       | 19        | Increased batch size (128 --> 256 --> 512)                   | Acc reduced                              |
| 10   | Ninth     | --           | 81.40              | 29       | 28        | Decrease batch size (128 --> 64)                             | Acc improved                             |
| 11   | Tenth     | --           | 82.13              | 29       | 41        | Decrease batch size (128 --> 64 --> 32)                      | Acc improved and crossed base (81.88%)   |



### Iterations

#### First

- Change Conv2D with SeparableConv2D
- No of parameters reduced to 604,213
- Val Acc is 100% at every epoch, it means model is learning nothing
- Plot is zig-zag with spikes

#### Second

- Add BN after every Relu layer
- No of parameters increased to 609,973
- Val Acc is 82.18%. Train Acc is 86.58. Diff - 4.4%
- Plot is zig-zag with small spikes
- Model started to converge better
- Still parameters are way high

#### Third

- No of parameters are high, need to bring it below 1,00,000
- Remove Dense layers : D1(393,728) and D2(131,328)
- Parameters after removing dense layer: 93,109. Within target
- Best Val Acc is 78.80%. Train Acc is 80.20%. Diff - 1.4%
-  loss: 0.5684 - acc: 0.8020 - val_loss: 0.9147 - val_acc: 0.7880 
- Plot is zig-zag with large spikes
- Modes started to overfit again and it started to diverge

#### Fourth

- Remove Dropout from last layer as it hurts performance if added in the last layer
- Parameters are same 
-  loss: 0.5209 - acc: 0.8168 - val_loss: 0.7110 - val_acc: 0.8050 
- Best Val Acc is 80.50%. Train Acc is 81.66%. Diff - 1.4%
- Overall:  loss: 0.4677 - acc: 0.8348 - val_loss: 0.7610 - val_acc: 0.7757 
- Plot is zig-zag with large spikes
- Model still overfitting

#### Fifth

- Reduce Dropout from 0.25 to 0.1
-  Parameters are same
- loss: 0.2618 - acc: 0.9041 - val_loss: 0.6139 - val_acc: 0.8092 
- Best Val Acc - 80.92%, Train Acc - 90.41%, Diff - 9.49
- Overall:  loss: 0.2468 - acc: 0.9099 - val_loss: 0.7971 - val_acc: 0.7614 
- Acc improved a little
- Plot is zig-zag with large spikes
- Model still overfitting

#### Sixth

- Remove all Dropout 
- Parameters are same
-  loss: 0.6459 - acc: 0.7797 - val_loss: 1.2350 - val_acc: 0.7100 
- Best Val Acc - 71.00%, Train Acc - 77.97%, Diff - 6.97
- Overall:  loss:  loss: 0.4373 - acc: 0.8498 - val_loss: 1.5209 - val_acc: 0.4979 
- Acc declined by a good amount
- Plot is zig-zag with small spikes
- Model still overfitting

#### Seventh

- Reset to Fifth(Dropout of 0.1)
- Increased batch size (128 --> 256)
- Parameters are same
- loss: 0.2981 - acc: 0.8912 - val_loss: 0.6302 - val_acc: 0.7991 
- Best Val Acc - 79.91%, Train Acc - 89.12%, Diff - 9.21
- Overall:  loss:  loss: 0.2577 - acc: 0.9068 - val_loss: 0.7243 - val_acc: 0.7808  
- Acc decreased a little
- Plot is zig-zag with large spikes
- Model still overfitting

#### Eighth

- Increased batch size (128 --> 256 --> 512)
- Parameters are same
-  loss: 0.2849 - acc: 0.8957 - val_loss: 0.6959 - val_acc: 0.7793 
- Best Val Acc - 77.93%, Train Acc - 89.57%, Diff -11.64
- Overall:   loss: 0.2681 - acc: 0.9030 - val_loss: 0.7287 - val_acc: 0.7772 
- Acc decreased a little
- Plot is zig-zag with large spikes
- Model still overfitting

#### Nineth

- Decrease batch size (128 --> 64)
- Parameters are same
- loss: 0.2962 - acc: 0.8922 - val_loss: 0.5841 - val_acc: 0.8140 
- Best Val Acc - 81.40%, Train Acc - 89.22%, Diff -7.82
- Overall:   loss: 0.2616 - acc: 0.9042 - val_loss: 0.6092 - val_acc: 0.8049 
- Acc increased 
- Plot is zig-zag with small spikes
- Model still overfitting

#### Tenth

- Decrease batch size (128 --> 64 --> 32)
- Parameters are same
-  Epoch 26 - loss: 0.4047 - acc: 0.8574 - val_loss: 0.5353 - val_acc: 0.8213 
- Best Val Acc - 82.13%, Train Acc - 85.74%, Diff -3.61
- Overall:   loss: 0.3035 - acc: 0.8910 - val_loss: 0.5662 - val_acc: 0.8138
- Acc increased 
- Plot is zig-zag with small spikes
- Model still overfitting
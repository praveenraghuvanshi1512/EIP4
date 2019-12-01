# EIP4 Assignment 3 - Praveen Raghuvanshi

#### Final Base Model Accuracy : 81.88%

```
Accuracy on test data is: 81.88
```

### Updated Model

- Val Accuracy(48th Epoch): **82.59%**
- Total parameters: **84,277**
- No of Epochs: 50
- ipynb file : [Solution Iteration Twelveth](EIP4_A3_Praveen_Raghuvanshi_82_44.ipynb)

```python
Accuracy on test data is: 81.60
```

##### Model 

```python
# Base Model : Define the model
# Replace Conv2D with SeparableConv2D
# Add BN before activation
# Remove D1(393,728) and D2(131,328)
# Remove Dropout from last layer 
# Reduce dropout (0.25 --> 0.1)
# Increase Batch size (128 --> 256)
# Reduce Batch size (128 --> 64)
# Reduce Batch size (128 --> 64 --> 32)
# Removed Last Dense layer --> Added GAP

model = Sequential()

model.add(SeparableConv2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3))) # O/P Size: 32x32x48 , RF:3X3
model.add(BatchNormalization()) 
model.add(Activation('relu')) 

model.add(SeparableConv2D(48, 3, 3)) # O/P Size: 32x32x48 , RF:5X5
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # O/P Size: 16x16x48 , RF:6X6
model.add(Dropout(0.1))

model.add(SeparableConv2D(96, 3, 3, border_mode='same')) # O/P Size: 16x16x96 , RF:10X10
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(96, 3, 3)) # O/P Size: 16x16x96 , RF:14X14
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # O/P Size: 8x8x96 , RF:16X16
model.add(Dropout(0.1))

model.add(SeparableConv2D(192, 3, 3, border_mode='same')) # O/P Size: 8x8x192 , RF:24X24
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(SeparableConv2D(192, 3, 3)) # O/P Size: 8x8x192 , RF:32X32
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) # O/P Size: 4x4x192 , RF:36X36
model.add(Dropout(0.1))

model.add(SeparableConv2D(num_classes, 4, border_mode='same')) # O/P Size: 4x4x10 , RF:52x52

model.add(GlobalAveragePooling2D())
model.add(Activation('softmax')) 

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
1562/1562 [==============================] - 53s 34ms/step - loss: 1.2843 - acc: 0.5367 - val_loss: 1.0842 - val_acc: 0.6185
Epoch 2/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.9188 - acc: 0.6764 - val_loss: 0.9258 - val_acc: 0.6803
Epoch 3/50
1562/1562 [==============================] - 49s 31ms/step - loss: 0.7881 - acc: 0.7249 - val_loss: 0.7868 - val_acc: 0.7305
Epoch 4/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.7133 - acc: 0.7516 - val_loss: 0.7924 - val_acc: 0.7330
Epoch 5/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.6624 - acc: 0.7684 - val_loss: 0.7295 - val_acc: 0.7463
Epoch 6/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.6140 - acc: 0.7858 - val_loss: 0.7141 - val_acc: 0.7584
Epoch 7/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.5857 - acc: 0.7949 - val_loss: 0.7703 - val_acc: 0.7485
Epoch 8/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.5592 - acc: 0.8041 - val_loss: 0.6813 - val_acc: 0.7710
Epoch 9/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.5268 - acc: 0.8161 - val_loss: 0.6261 - val_acc: 0.7870
Epoch 10/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.5070 - acc: 0.8227 - val_loss: 0.5891 - val_acc: 0.8015
Epoch 11/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4845 - acc: 0.8298 - val_loss: 0.6300 - val_acc: 0.7896
Epoch 12/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4668 - acc: 0.8362 - val_loss: 0.7000 - val_acc: 0.7726
Epoch 13/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4522 - acc: 0.8406 - val_loss: 0.6797 - val_acc: 0.7741
Epoch 14/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4392 - acc: 0.8450 - val_loss: 0.7646 - val_acc: 0.7599
Epoch 15/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4216 - acc: 0.8513 - val_loss: 0.6265 - val_acc: 0.7958
Epoch 16/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.4107 - acc: 0.8561 - val_loss: 0.6135 - val_acc: 0.8053
Epoch 17/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3963 - acc: 0.8602 - val_loss: 0.6659 - val_acc: 0.7826
Epoch 18/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3860 - acc: 0.8635 - val_loss: 0.6520 - val_acc: 0.7912
Epoch 19/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3738 - acc: 0.8682 - val_loss: 0.6832 - val_acc: 0.7843
Epoch 20/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3658 - acc: 0.8696 - val_loss: 0.6009 - val_acc: 0.8066
Epoch 21/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3572 - acc: 0.8720 - val_loss: 0.6503 - val_acc: 0.8064
Epoch 22/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3457 - acc: 0.8778 - val_loss: 0.5709 - val_acc: 0.8147
Epoch 23/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3381 - acc: 0.8791 - val_loss: 0.6083 - val_acc: 0.8096
Epoch 24/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3336 - acc: 0.8824 - val_loss: 0.5695 - val_acc: 0.8207
Epoch 25/50
1562/1562 [==============================] - 48s 30ms/step - loss: 0.3237 - acc: 0.8850 - val_loss: 0.5997 - val_acc: 0.8168
Epoch 26/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3153 - acc: 0.8887 - val_loss: 0.6003 - val_acc: 0.8090
Epoch 27/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3136 - acc: 0.8894 - val_loss: 0.6294 - val_acc: 0.8115
Epoch 28/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3028 - acc: 0.8917 - val_loss: 0.6110 - val_acc: 0.8114
Epoch 29/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.3016 - acc: 0.8936 - val_loss: 0.7256 - val_acc: 0.7877
Epoch 30/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2916 - acc: 0.8962 - val_loss: 0.5975 - val_acc: 0.8151
Epoch 31/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2868 - acc: 0.8982 - val_loss: 0.6200 - val_acc: 0.8110
Epoch 32/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2880 - acc: 0.8968 - val_loss: 0.6016 - val_acc: 0.8205
Epoch 33/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2764 - acc: 0.9006 - val_loss: 0.6457 - val_acc: 0.8112
Epoch 34/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2742 - acc: 0.9009 - val_loss: 0.5811 - val_acc: 0.8229
Epoch 35/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2652 - acc: 0.9052 - val_loss: 0.5879 - val_acc: 0.8188
Epoch 36/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2627 - acc: 0.9064 - val_loss: 0.6543 - val_acc: 0.8102
Epoch 37/50
1562/1562 [==============================] - 48s 30ms/step - loss: 0.2631 - acc: 0.9044 - val_loss: 0.6193 - val_acc: 0.8161
Epoch 38/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2582 - acc: 0.9084 - val_loss: 0.6099 - val_acc: 0.8106
Epoch 39/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2498 - acc: 0.9100 - val_loss: 0.5843 - val_acc: 0.8222
Epoch 40/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2515 - acc: 0.9098 - val_loss: 0.6101 - val_acc: 0.8145
Epoch 41/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2453 - acc: 0.9120 - val_loss: 0.6001 - val_acc: 0.8219
Epoch 42/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2443 - acc: 0.9106 - val_loss: 0.6396 - val_acc: 0.8184
Epoch 43/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2391 - acc: 0.9140 - val_loss: 0.6506 - val_acc: 0.8141
Epoch 44/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2419 - acc: 0.9129 - val_loss: 0.5826 - val_acc: 0.8245
Epoch 45/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2298 - acc: 0.9173 - val_loss: 0.6746 - val_acc: 0.8042
Epoch 46/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2325 - acc: 0.9157 - val_loss: 0.6638 - val_acc: 0.8100
Epoch 47/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2285 - acc: 0.9175 - val_loss: 0.6530 - val_acc: 0.8168
Epoch 48/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2237 - acc: 0.9200 - val_loss: 0.6083 - val_acc: 0.8259
Epoch 49/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2189 - acc: 0.9219 - val_loss: 0.6156 - val_acc: 0.8244
Epoch 50/50
1562/1562 [==============================] - 48s 31ms/step - loss: 0.2204 - acc: 0.9198 - val_loss: 0.6606 - val_acc: 0.8160
Model took 2403.04 seconds to train

Accuracy on test data is: 81.60
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
| 8    | Seventh   | --           | 69.68              | 27       | 12.9      | Brought Dropout back. Fifth                                  | Acc reduced                              |
| 9    | Eighth    | --           | 79.91              | 29       | 20        | Increase Batch size (128 --> 256)                            | Acc improved                             |
| 10   | Ninth     | --           | 77.93              | 29       | 19        | Increased batch size (128 --> 256 --> 512)                   | Acc reduced                              |
| 11   | Tenth     | --           | 81.40              | 29       | 28        | Decrease batch size (128 --> 64)                             | Acc improved                             |
| 12   | Eleventh  | --           | 82.13              | 29       | 41        | Decrease batch size (128 --> 64 --> 32)                      | Acc improved and crossed base (81.88%)   |
| 13   | Twelveth  | 84,277       | 82.44              |          | 38        | Last Dense and Flatten layer replaced with GAP               | Parameters reduced, Acc improved         |
| 14   | Solution  | --           | 82.59              |          | 38        | --                                                           | --                                       |



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
- Parameters are same
- loss: loss: 0.7627 - acc: 0.7368 - val_loss: 1.2213 - val_acc: 0.6968
- Best Val Acc - 69.68%, Train Acc - 73.68%, Diff - 4
- Overall:  loss:  loss: 0.4373 - acc: 0.8498 - val_loss: 1.5209 - val_acc: 0.4979
- Acc decreased a little
- Plot is zig-zag with large spikes
- Model still overfitting

#### Eighth

- Reset to Fifth(Dropout of 0.1)
- Increased batch size (128 --> 256)
- Parameters are same
- loss: 0.2752 - acc: 0.8981 - val_loss: 0.6591 - val_acc: 0.7936
- Best Val Acc - 79.36%, Train Acc - 89.81%, Diff - 10.451
- Overall:  loss: 0.2577 - acc: 0.9068 - val_loss: 0.7243 - val_acc: 0.7808 
- Acc improved 
- Plot is zig-zag with large spikes
- Model still overfitting

#### Ninth

- Increased batch size (128 --> 256 --> 512)
- Parameters are same
- loss: 0.2849 - acc: 0.8957 - val_loss: 0.6959 - val_acc: 0.7793 
- Best Val Acc - 77.93%, Train Acc - 89.57%, Diff -11.64
- Overall:   loss: 0.2681 - acc: 0.9030 - val_loss: 0.7287 - val_acc: 0.7772 
- Acc decreased a little
- Plot is zig-zag with large spikes
- Model still overfitting

#### Tenth

- Decrease batch size (128 --> 64)
- Parameters are same
- loss: 0.2962 - acc: 0.8922 - val_loss: 0.5841 - val_acc: 0.8140 
- Best Val Acc - 81.40%, Train Acc - 89.22%, Diff -7.82
- Overall:   loss: 0.2616 - acc: 0.9042 - val_loss: 0.6092 - val_acc: 0.8049 
- Acc increased 
- Plot is zig-zag with small spikes
- Model still overfitting

#### Eleventh

- Decrease batch size (128 --> 64 --> 32)
- Parameters are same
-  Epoch 26 - loss: 0.4047 - acc: 0.8574 - val_loss: 0.5353 - val_acc: 0.8213 
- Best Val Acc - 82.13%, Train Acc - 85.74%, Diff -3.61
- Overall:   loss: 0.3035 - acc: 0.8910 - val_loss: 0.5662 - val_acc: 0.8138
- Acc increased 
- Plot is zig-zag with small spikes
- Model still overfitting

#### Twelveth

- Removed last Dense layer
- Added GAP
- Parameters reduced to 84,277
- Epoch 26 - loss: 0.2179 - acc: 0.9205 - val_loss: 0.6110 - val_acc: 0.8244
- Best Val Acc - 82.44%, Train Acc - 92.05%, Diff -9.61
- Overall:   same as above
- Acc increased 
- Plot is zig-zag with small spikes
- Model still overfitting
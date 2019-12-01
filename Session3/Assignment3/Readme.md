# EIP4 Assignment 3 - Praveen Raghuvanshi

#### Final Base Model Accuracy : 81.88%

```
Accuracy on test data is: 81.88
```

### Updated Model

- Val Accuracy(49th Epoch): **83.98%**
- Total parameters: **84,277**
- No of Epochs: 50
- ipynb file : [Solution](EIP4_A3_Praveen_Raghuvanshi_Detailed.ipynb)

```python
Accuracy on test data is: 82.31
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
# Removed Last Dense layer --> Added GAP
# Image Augmentation (Rotation and flip)
# Image Augmentation (Rotation- 90 and horizontal flip)
# Image Augmentation (Rotation range, horizontal flip, shift(width and height))

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

```python
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.where in 2.0, which has the same broadcast rule as np.where
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
  
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:14: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=(array([[[..., verbose=1, steps_per_epoch=1562, epochs=50)`
  
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.

WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.

Epoch 1/50
1562/1562 [==============================] - 61s 39ms/step - loss: 1.4158 - acc: 0.4871 - val_loss: 1.2780 - val_acc: 0.5644
Epoch 2/50
1562/1562 [==============================] - 56s 36ms/step - loss: 1.0960 - acc: 0.6101 - val_loss: 1.1510 - val_acc: 0.6210
Epoch 3/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.9708 - acc: 0.6593 - val_loss: 1.0909 - val_acc: 0.6316
Epoch 4/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.8982 - acc: 0.6877 - val_loss: 0.8331 - val_acc: 0.7111
Epoch 5/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.8404 - acc: 0.7071 - val_loss: 0.8924 - val_acc: 0.6952
Epoch 6/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.8029 - acc: 0.7193 - val_loss: 0.8034 - val_acc: 0.7297
Epoch 7/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.7739 - acc: 0.7289 - val_loss: 0.9305 - val_acc: 0.6924
Epoch 8/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.7459 - acc: 0.7381 - val_loss: 0.8604 - val_acc: 0.7066
Epoch 9/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.7248 - acc: 0.7478 - val_loss: 0.8059 - val_acc: 0.7354
Epoch 10/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.7080 - acc: 0.7553 - val_loss: 0.7535 - val_acc: 0.7423
Epoch 11/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.6871 - acc: 0.7603 - val_loss: 0.7263 - val_acc: 0.7648
Epoch 12/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.6772 - acc: 0.7649 - val_loss: 0.6379 - val_acc: 0.7788
Epoch 13/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.6638 - acc: 0.7704 - val_loss: 0.6586 - val_acc: 0.7713
Epoch 14/50
1562/1562 [==============================] - 57s 37ms/step - loss: 0.6445 - acc: 0.7737 - val_loss: 0.6197 - val_acc: 0.7895
Epoch 15/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.6360 - acc: 0.7800 - val_loss: 0.6554 - val_acc: 0.7772
Epoch 16/50
1562/1562 [==============================] - 57s 37ms/step - loss: 0.6251 - acc: 0.7824 - val_loss: 0.7026 - val_acc: 0.7596
Epoch 17/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.6179 - acc: 0.7844 - val_loss: 0.6000 - val_acc: 0.7966
Epoch 18/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.6077 - acc: 0.7881 - val_loss: 0.5734 - val_acc: 0.8041
Epoch 19/50
1562/1562 [==============================] - 59s 38ms/step - loss: 0.5980 - acc: 0.7915 - val_loss: 0.5739 - val_acc: 0.8027
Epoch 20/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.5929 - acc: 0.7934 - val_loss: 0.6599 - val_acc: 0.7753
Epoch 21/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5893 - acc: 0.7932 - val_loss: 0.5685 - val_acc: 0.8044
Epoch 22/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.5792 - acc: 0.7993 - val_loss: 0.5530 - val_acc: 0.8127
Epoch 23/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.5700 - acc: 0.8012 - val_loss: 0.5730 - val_acc: 0.8065
Epoch 24/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5585 - acc: 0.8070 - val_loss: 0.5115 - val_acc: 0.8272
Epoch 25/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.5633 - acc: 0.8052 - val_loss: 0.6537 - val_acc: 0.7806
Epoch 26/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5533 - acc: 0.8082 - val_loss: 0.6639 - val_acc: 0.7823
Epoch 27/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5492 - acc: 0.8108 - val_loss: 0.5693 - val_acc: 0.8067
Epoch 28/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5409 - acc: 0.8113 - val_loss: 0.6573 - val_acc: 0.7879
Epoch 29/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5371 - acc: 0.8132 - val_loss: 0.5898 - val_acc: 0.7990
Epoch 30/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.5374 - acc: 0.8128 - val_loss: 0.6259 - val_acc: 0.7976
Epoch 31/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5314 - acc: 0.8159 - val_loss: 0.6174 - val_acc: 0.7936
Epoch 32/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5270 - acc: 0.8169 - val_loss: 0.6259 - val_acc: 0.7969
Epoch 33/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5191 - acc: 0.8190 - val_loss: 0.6012 - val_acc: 0.8010
Epoch 34/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5207 - acc: 0.8200 - val_loss: 0.6095 - val_acc: 0.7975
Epoch 35/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.5149 - acc: 0.8210 - val_loss: 0.5695 - val_acc: 0.8165
Epoch 36/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5079 - acc: 0.8230 - val_loss: 0.6296 - val_acc: 0.7922
Epoch 37/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5064 - acc: 0.8239 - val_loss: 0.6008 - val_acc: 0.8033
Epoch 38/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5079 - acc: 0.8239 - val_loss: 0.5133 - val_acc: 0.8283
Epoch 39/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5007 - acc: 0.8254 - val_loss: 0.5393 - val_acc: 0.8218
Epoch 40/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.5019 - acc: 0.8256 - val_loss: 0.5544 - val_acc: 0.8176
Epoch 41/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.4909 - acc: 0.8293 - val_loss: 0.5348 - val_acc: 0.8200
Epoch 42/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.4895 - acc: 0.8298 - val_loss: 0.5786 - val_acc: 0.8151
Epoch 43/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.4879 - acc: 0.8290 - val_loss: 0.5414 - val_acc: 0.8225
Epoch 44/50
1562/1562 [==============================] - 58s 37ms/step - loss: 0.4880 - acc: 0.8312 - val_loss: 0.5438 - val_acc: 0.8190
Epoch 45/50
1562/1562 [==============================] - 57s 37ms/step - loss: 0.4816 - acc: 0.8321 - val_loss: 0.4910 - val_acc: 0.8350
Epoch 46/50
1562/1562 [==============================] - 56s 36ms/step - loss: 0.4815 - acc: 0.8327 - val_loss: 0.5055 - val_acc: 0.8333
Epoch 47/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.4791 - acc: 0.8342 - val_loss: 0.6006 - val_acc: 0.8118
Epoch 48/50
1562/1562 [==============================] - 57s 37ms/step - loss: 0.4690 - acc: 0.8361 - val_loss: 0.5417 - val_acc: 0.8258
Epoch 49/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.4694 - acc: 0.8373 - val_loss: 0.4833 - val_acc: 0.8398
Epoch 50/50
1562/1562 [==============================] - 57s 36ms/step - loss: 0.4676 - acc: 0.8370 - val_loss: 0.5423 - val_acc: 0.8231
Model took 2842.91 seconds to train

Accuracy on test data is: 82.31
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

    

| S.No | Iteration       | # Parameters | Val Acc (Best)     | # layers | Time(min) | Model Changes                                                | Remark                                   |
| ---- | --------------- | ------------ | ------------------ | -------- | --------- | ------------------------------------------------------------ | ---------------------------------------- |
| 1    | Base            | 1,172, 410   | 81.88 (50th Epoch) | 26       | 6.2       | No Change                                                    | N/A                                      |
| 2    | Solution(Final) | 84,277       | 83.98(Best)        |          | 47        | Refer Iteration# 15                                          | Best acccuracy                           |
| 3    | First           | 604,213      | 100                | 26       | 10        | Conv2D --> SeparableConv2D                                   | Parameters reduced, Acc is constant 1.00 |
| 4    | Second          | 609,973      | 82.18              | 32       | 12.8      | Batch normalization (SepConv -> BN -> ReLU -> SepConv -> BN -> ReLU -> MP) | Parameters increased, Accuracy is normal |
| 5    | Third           | 93,109       | 78.80              | 30       | 12.9      | Remove Dense Layer D1(393,728) and D2(131,328)               | Parameters is under 100,000, Acc reduced |
| 6    | Fourth          | --           | 80.50              | 29       | 12.9      | Remove Dropout from last layer                               | Acc improved                             |
| 7    | Fifth           | --           | 80.92              | 29       | 12.9      | Reduce Dropout from 0.25 to 0.1                              | Acc improved                             |
| 8    | Sixth           | --           | 78.98              | 27       | 12.9      | Remove all Dropout                                           | Acc reduced                              |
| 9    | Seventh         | --           | 69.68              | 27       | 12.9      | Brought Dropout back. Fifth                                  | Acc reduced                              |
| 10   | Eighth          | --           | 79.91              | 29       | 20        | Increase Batch size (128 --> 256)                            | Acc improved                             |
| 11   | Ninth           | --           | 77.93              | --       | 19        | Increased batch size (128 --> 256 --> 512)                   | Acc reduced                              |
| 12   | Tenth           | --           | 81.40              | --       | 28        | Decrease batch size (128 --> 64)                             | Acc improved                             |
| 13   | Eleventh        | --           | 82.13              | --       | 41        | Decrease batch size (128 --> 64 --> 32)                      | Acc improved and crossed base (81.88%)   |
| 14   | Twelveth        | 84,277       | 82.44              |          | 38        | Last Dense and Flatten layer replaced with GAP               | Parameters reduced, Acc improved         |
| 15   | Thirteenth      | --           | 69.48              |          | 43        | Image Augmentation (horizontal and vertical flip, rotation)  | Acc reduced drastically to 69.48         |
| 16   | Fourteenth      | --           |                    |          |           |                                                              |                                          |
| 17   | Fifteenth       | --           | 83.11              |          | 30        | Image Augmentation(horizontal flip, rotation range, slide(height and width)) | Acc improved with best accuracy          |



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

#### Thirteenth

- Image augmentation 
  - Horizontal and vertical flip
  - rotation : 90
- Epoch 26 - loss: loss: 0.9147 - acc: 0.6798 - val_loss: 0.8799 - val_acc: 0.6948
- Best Val Acc - 69.48%, Train Acc - 67.98%, Diff -2
- Overall:   loss: 0.8432 - acc: 0.7061 - val_loss: 0.9430 - val_acc: 0.6782
- Acc reduced drastically
- Plot is zig-zag with small spikes
- Model still overfitting

#### Fourteenth

- Image augmentation 
  - Horizontal flip
  - rotation : 90
- Epoch 41- loss: 0.7552 - acc: 0.7370 - val_loss: 0.7107 - val_acc: 0.7585
- Best Val Acc - 75.85%, Train Acc - 73.70%, Diff -2.15
- Overall: loss: 0.7311 - acc: 0.7473 - val_loss: 0.7469 - val_acc: 0.7463
- Acc reduced drastically
- Plot is zig-zag with small spikes, converging better with image augmentation
- Model still overfitting

#### Fifteenth - Solution(83.98%)

- Looks like rotation by 90 degree is an overkill.

- Image augmentation 

  ```python
  rotation_range=15,
  horizontal_flip=True,
  width_shift_range=0.1,
  height_shift_range=0.1
  ```

- Epoch 50- loss: 0.4645 - acc: 0.8396 - val_loss: 0.5118 - val_acc: 0.8311

- Best Val Acc - 83.11%, Train Acc - 83.96%, Diff -0.85

- Overall: same as above

- Acc improved with best accuracy

- Plot is zig-zag with small spikes, converging better with image augmentation
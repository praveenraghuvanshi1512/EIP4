# EIP4 Assignment 2 - Praveen Raghuvanshi

### Model Test Accuracy - 99.44%

#### No of Total Parameters : 14,720

#### ipynb file : [Solution](EIP4_A2_Praveen_Raghuvanshi.ipynb)

#### Result

```python
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
```

```python
[0.019735304067994, 0.9944]
```

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 14s 229us/step - loss: 0.5443 - acc: 0.8466 - val_loss: 0.0999 - val_acc: 0.9805
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 11s 180us/step - loss: 0.2610 - acc: 0.9220 - val_loss: 0.0545 - val_acc: 0.9888
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 11s 187us/step - loss: 0.2039 - acc: 0.9389 - val_loss: 0.0509 - val_acc: 0.9872
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1732 - acc: 0.9455 - val_loss: 0.0373 - val_acc: 0.9914
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1544 - acc: 0.9481 - val_loss: 0.0396 - val_acc: 0.9893
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 11s 180us/step - loss: 0.1431 - acc: 0.9502 - val_loss: 0.0322 - val_acc: 0.9909
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1342 - acc: 0.9514 - val_loss: 0.0356 - val_acc: 0.9905
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 11s 176us/step - loss: 0.1283 - acc: 0.9521 - val_loss: 0.0271 - val_acc: 0.9929
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 11s 179us/step - loss: 0.1229 - acc: 0.9535 - val_loss: 0.0231 - val_acc: 0.9938
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 11s 183us/step - loss: 0.1159 - acc: 0.9539 - val_loss: 0.0238 - val_acc: 0.9932
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 11s 177us/step - loss: 0.1117 - acc: 0.9551 - val_loss: 0.0253 - val_acc: 0.9927
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 11s 177us/step - loss: 0.1136 - acc: 0.9530 - val_loss: 0.0225 - val_acc: 0.9944
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 11s 182us/step - loss: 0.1077 - acc: 0.9553 - val_loss: 0.0239 - val_acc: 0.9934
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1050 - acc: 0.9572 - val_loss: 0.0222 - val_acc: 0.9941
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 11s 184us/step - loss: 0.1013 - acc: 0.9571 - val_loss: 0.0208 - val_acc: 0.9940
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1003 - acc: 0.9582 - val_loss: 0.0207 - val_acc: 0.9937
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 11s 179us/step - loss: 0.0992 - acc: 0.9576 - val_loss: 0.0196 - val_acc: 0.9944
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 11s 178us/step - loss: 0.1008 - acc: 0.9552 - val_loss: 0.0231 - val_acc: 0.9942
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 11s 180us/step - loss: 0.0976 - acc: 0.9564 - val_loss: 0.0195 - val_acc: 0.9946
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0964 - acc: 0.9567 - val_loss: 0.0197 - val_acc: 0.9944
<keras.callbacks.History at 0x7f3b71922ef0>
```



##### Analysis

- In [EIGHTH](https://tinyurl.com/y494cv85) Solution, we could have an accuracy of 99.47% which is more than required for this assignment. However, no of parameters (16,604) is above than the required of 15,000. 
- First objective was to reduce the no of parameters and bring it under 15,000 and second was to improve accuracy within 20 epochs
- As per the assignment rule, we are not supposed to use bias. Added use_bias=False in all the convolution layers. This reduced the parameters little and brought down to 16,472. Accuracy reduced to 99.09%
- As we were still above the parameters threshold of 150000 by 472, we needed a way to further to reduce the parameters.
- On analyzing SEVENTH and EIGHTH solution, there was a reduction in number of parameters from SEVENTH(18,326) to EIGHTH(16,604) solution with  a change of no of filters and dropout.
  - Dropout was reduced from 0.25 to 0.1. Dropout doesn't change number of parameter. So we retained it to 0.1.
  - No of filters in convolution layers
    - SEVENTH solution has a sequence Conv1(10) -> Conv2(16) -> Conv3(32) -> MP -> Conv4(10) -> Conv5(16) -> Conv6(32) -> Conv7(10) -> Conv8(10) .
      - The pattern is 10-16-32-MP-10-16-32-10-10
    - EIGHTH solution has a sequence Conv1(16) -> Conv2(32) -> Conv3(10) -> MP -> Conv4(16) -> Conv5(16) -> Conv6(16) -> Conv7(16) -> Conv8(10) .
      - The pattern is 16-32-10-MP-16-16-16-16-10
    - On comparing SEVENTH and EIGHTH solution, we could see after Max Pooling, no of filters have been reduced leading to less no of parameters. 
    - We need to reduce the parameters, so we also reduced the no of filters for some layers in our model and brought down the no of parameters.
    - The solution has 14,720 parameters with an accuracy of 99.44% and trained for 20 epochs. The sequence of layers is as follows:  Conv1(16) -> Conv2(32) -> Conv3(10) -> MP -> Conv4(16) -> Conv5(16) -> Conv6(10) -> Conv7(16) -> Conv8(10) .
      - The pattern is 16-32-10-MP-16-10-16-16-10
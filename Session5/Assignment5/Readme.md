# EIP4 Assignment 5 - Praveen Raghuvanshi

## Accuracy

```python
{
    'age_output_acc': 0.42224702380952384,
    'bag_output_acc': 0.5569196428571429,
    'emotion_output_acc': 0.6934523809523809,
    'footwear_output_acc': 0.6290922619047619,
    'gender_output_acc': 0.8497023809523809,
    'image_quality_output_acc': 0.5654761904761905,
    'loss': 14.37654549734933,
    'pose_output_acc': 0.8098958333333334,
    'weight_output_acc': 0.6369047619047619
}
```

#### Last Epoch(200)

```python
Epoch 200/200 339/339 [==============================] - 114s 337ms/step - loss: 1.6997 - gender_output_loss: 0.0263 - image_quality_output_loss: 0.2859 - age_output_loss: 0.3264 - weight_output_loss: 0.3732 - bag_output_loss: 0.1355 - footwear_output_loss: 0.1492 - pose_output_loss: 0.0567 - emotion_output_loss: 0.3465 - gender_output_acc: 0.9925 - image_quality_output_acc: 0.8945 - age_output_acc: 0.8703 - weight_output_acc: 0.8530 - bag_output_acc: 0.9541 - footwear_output_acc: 0.9546 - pose_output_acc: 0.9842 - emotion_output_acc: 0.8476 - val_loss: 14.3765 - val_gender_output_loss: 0.8426 - val_image_quality_output_loss: 1.5006 - val_age_output_loss: 3.2410 - val_weight_output_loss: 1.6227 - val_bag_output_loss: 2.3991 - val_footwear_output_loss: 1.6992 - val_pose_output_loss: 1.0435 - val_emotion_output_loss: 2.0278 - **val_gender_output_acc: 0.8497** - val_image_quality_output_acc: 0.5655 - val_age_output_acc: 0.4222 - val_weight_output_acc: 0.6369 - val_bag_output_acc: 0.5569 - val_footwear_output_acc: 0.6291 - val_pose_output_acc: 0.8099 - val_emotion_output_acc: 0.6935
```

 

#### Best Epoch(102) - Gender

```python
Epoch 102/200 339/339 [==============================] - 115s 341ms/step - loss: 3.5252 - gender_output_loss: 0.0433 - image_quality_output_loss: 0.9074 - age_output_loss: 0.5733 - weight_output_loss: 0.6551 - bag_output_loss: 0.2548 - footwear_output_loss: 0.3525 - pose_output_loss: 0.0709 - emotion_output_loss: 0.6680 - gender_output_acc: 0.9881 - image_quality_output_acc: 0.5918 - age_output_acc: 0.7783 - weight_output_acc: 0.7582 - bag_output_acc: 0.9039 - footwear_output_acc: 0.8498 - pose_output_acc: 0.9806 - emotion_output_acc: 0.7548 - val_loss: 8.8914 - val_gender_output_loss: 0.5512 - val_image_quality_output_loss: 0.9589 - val_age_output_loss: 2.0237 - val_weight_output_loss: 1.1452 - val_bag_output_loss: 1.4128 - val_footwear_output_loss: 1.1776 - val_pose_output_loss: 0.7155 - val_emotion_output_loss: 0.9065 - **val_gender_output_acc: 0.8568** - val_image_quality_output_acc: 0.5815 - val_age_output_acc: 0.4152 - val_weight_output_acc: 0.6440 - val_bag_output_acc: 0.6097 - val_footwear_output_acc: 0.6834 - val_pose_output_acc: 0.8125 - val_emotion_output_acc: 0.7295 
```



- Best Val Accuracy - Gender(102th Epoch): **85.68%**
- Total parameters: **7,103,222**
- No of Epochs: **200**
- Network: **DenseNet121**
- ipynb file : [Solution](EIP4_A5_Praveen_Raghuvanshi_DenseNet_Final.ipynb)

Note: In case text in ipynb file is truncated, please open github URL in [https://nbviewer.jupyter.org](https://nbviewer.jupyter.org/)
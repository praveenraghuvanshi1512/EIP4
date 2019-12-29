# EIP4 - Session 5 : 

#### Video 

06-Dec-2019:  https://www.youtube.com/watch?v=msmv7xM7JiA 

[![EIP4 Session 2](http://img.youtube.com/vi/77YpTc684ls/0.jpg)](https://youtu.be/77YpTc684ls)	

07-Dec-2019: https://youtu.be/2_PS0jWPf-4

[![EIP4 Session 5 Saturday](http://img.youtube.com/vi/2_PS0jWPf-4/0.jpg)](https://youtu.be/2_PS0jWPf-4)

#### Resources

- [Receptive Field](https://distill.pub/2019/computing-receptive-fields/)
- 

#### Notes

##### Quiz

- Loss is always between 0 and 1 and NOT 0 OR 1. 
- SGD with Momentum is the best algorithm

##### Session

- 1x1 is known as point-wise convolution

- No of pixels, we are directly looking is know as Local Receptive Field(LRF) and the ones we are looking indirectly is known as Global Receptive Field(GRF)

- Strides are not good for us

  - It helps increase RF by a good margin
  - In case of MP, we keep things in memory and proceed
  - Stride is mainly used to decrease size of image drastically similar to MP
  - Learning will be less compared to normal convolution
  - Strides creates a checkerboard issue. In checkerboard we are using 2 pixel at a time
  - We should use strides to reduce no of layers in a hardware constrained data
  -  

  ![Stride vs MP](.\assets\Stride_vs_MP.JPG)

- Dialated convolution

  - When we want to retain the size of image btw input and output (1080 x 1080 --> 1080 x 1080)
  - It's mainly required in Medical boundaries where resolution of output is of great importance
  - We need to know each pixel belong to which class
  - A dialated kernel is never used alone, it'll always be used with 3x3
  - We add dialated pixels in the image based on the output expected. It can be 1,2, 3... etc. 
  - The value of dialated pixel could be sum/average of neighboring pixels. 
  - We are taking an image, scaling and convolving on top of that.
  - This is not deconvolution. 
  - This is transposed convolution
  - Whenever there is a uneven distribution information of pixel, we see a checkerboard issue.
  - Pixel shuffle is the solution to checkerboard issue
  - Use Depthwise separable convolution with 3x3 followed by 1x1 for assignment

- Depthwise separable convolution

  - We use 9 times more parameters in normal convolution compared to depthwise.

  - Its used in resource constraint hardware.

  - We use 1x1 to INCREASE no of channels in case of depthwise convolution or something similar.

  - All mobile networks are using this depthwise convolution due to less no of parameters

  - 1 kernel is looking at 1 channel only and not all the channels in normal case

  - Inception, v2 uses algorithm heavily

  - Depthwise convolution is not as strong as 3x3 convolution

  - We use Resnet(Microsoft) compared to Inception(Google) even though Inception is better. Reason being Resnet is much familiar/simpler

  - In mobile phones google is following Microsoft's Resnet due to simplicity.

  - Depthwise separable convolution is very fast and good but used only in mobiles.

    ![Depthwise convolution](.\assets\Depthwise_convolution.JPG)

- Concatenation and Addition of layers

- Batch Normalization

  - Take image of a white dog in sunlight(afternoon) and in dark(night)

  - A white dog may appear dark in a picture taken in dark

  - Are we going to have two kernels or one for this differentiation?

  - We are going to have only ONE kernel

  - This inconsistency is filled by normalization.

  - Normalization, we are changing the scale

  - There is not change in data

    ![Normalization](.\assets\Normalization.JPG)

  - ![Normalization curve](.\assets\Normalization_curve.JPG)

  - Normalization removes bias such as for a nose in the image of a cat. model will predict it as Nose for both white and black cat.

  - Normalization improves amplitude of features.

  - Amplitude scale up very fast in DNN

  - Before BN, we were not able to go beyond 15, 16 layers due to complex amplitude.

  - With BN, we were able to move to 120 layers and beyond

  - Make some changes in network and allow it to undo as well

  - Any addition/subraction(+/-) leads to shifting of data

  - Any Multiplication/Division(*/) leads to scaling of data

    ![Normalization equation](.\assets\Normalization_equation.JPG)

    ![Batch Normalization Accuracy](.\assets\BatchNormalizaiton_Acc.JPG)

  - Add BN to layers except last one

  - It doesn't matter if you apply ReLU / BN first

- Regularization

  - Add a dropout(0.25) after last layer and check the training accuracy. Run val acc immediately. Val acc will always be high.
  - Cover weight decay if time is there
  - Use small regularization one by one
  - See diagnosis and add that

- Receptive Field

  ![Receptive Field Calculation](.\assets\ReceptiveField_Calculation.JPG)

- Assignment
  - There is no BN
  
  - Proper dropout values are small values
  
  - Regularization must be low
  
  - Use [SeparableConv2D]( https://www.tensorflow.org/api_docs/python/tf/keras/layers/SeparableConv2D)
  
    ```python
    keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
    ```
  
  - Thigs that can be tried 
  
    - Replace Conv2D with SeparableConv2D - Reduce parameter
    - Add BN after every layer - Reduce Parameter
    - Use Bottleneck

#### Clarifications

- What is backpropagation? How weights are modified in backpropagation?
- How to choose batch size? How it affects time and accuracy?
- Why strides are not good ?
- There are different ways of reducing the parameters(1x1, depthwise, reduce no of filters). When to use them?
- Concatenation and Addition of layers
- What is mini-batch in BN?



### Resources

- [Freeze layers - Keras Tutorial : Fine-tuning using pre-trained models](https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/)


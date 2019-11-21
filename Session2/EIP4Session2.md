# EIP4 - Session 2 : 18-Nov-2019

#### Video 

 https://www.youtube.com/watch?v=33bBsEtNBOc

[![EIP4 Session 2](http://img.youtube.com/vi/33bBsEtNBOc/0.jpg)](http://www.youtube.com/watch?v=33bBsEtNBOc)





#### Notes

##### Quiz

- 1x1 is not a feature extractor

- 1x1 is a feature merger 

- 2x2 is mostly used in MP

- Why 2x2 for MP

  - Consider an image of 6x6 (36)
    - On MP with 3x3, we get output of 2x2(4) Reduction by 4/36 ~ 9%
    - On MP with 2x2, we get output of 3x3(9) Reduction by 9/36 ~ 25%
  - 25% reduction is better than 9%
  - MP is already reducing information a lot
  - With 9% there is hardly any pixel left
  - Never saw MP with stride of 3.
  - Have seen MP with stride of 2

- Kernel doesn't have RF, Channel does.

- Why 512 before MP?

  - It depends on the image size and hardware.

  - We were working on 400x400 which is a quite large image, so we can go till 512.

  - This can be bumped to larger values such as 1024, 2048 in case of Medical images where 

    accuracy and precision is important. 

  - MNIST has small images of 28x28 and we need not go till 512 for it

  - Small image may not have much information such as texture and edges

  - This no needs to be small for smaller images/constrained H/W

- Max Pooling 195x195x512 | ?x?x512x32

  - It could be 3x3
  - Replaced with 1x1 leading to reduction in parameters and faster also.

- -ve number in last layer signifies its not deterimental. For e.g if we get a tyre with negative value, it means it's NOT a dog

- Most of the numbers will be between -3 and +3

- Last layer won't be a Transition block, it'll be output layer

- When we are close to output, no need for MP.

- Initially network doesn't know things and slowly starts to loose information that is of no use in order to make sense.

- 32 -> 64 -> 128 -> 256 -> 512 --------> 32 -> 64 -> 128 -> 256 -> 512 

- Activation Functions

  - Neuron in our brain is capable of making a decision of whether an information needs to be sent further or not.
  - During convolution, we are take all the information and sending it further with no restriction
  - Initially Sigmoid was used since 1950's
  - Problem with sigmoid is it gives values between 0 and 1 finally.
  - Never going to use Sigmoid
  - Tanh introduced to include -ve values -1 to +1
  - ReLU only allows +ve numbers to go forward
  - Selu, Elu, Pelu

- First bit in a number tells sign(+ve or -ve)

  - ![Sign of a number](.\assets\Number_Mantisa_sign.JPG)
  - Checking 1 bit is cheap and fast
  - Send this bit to Relu. Relu only checks 1 bit
  - For any activation Fn, take this bit and send it to Swish which will be much accurate
  - Activation Fn other than Relu checks for more bits leading to huge calculation.
  - Relu is preferred for low and fast computation
  - 80% Relu and 20% LeakyRelu for academics
  - SGD with momentum best combo. Its state of art used in Resnet, Densenet, VGG16, Alexnet
  - 

- ![Batch Normalization](.\assets\Batch_Normalization.JPG)

  - X : No
  - Mu : Shift
  - Sigma, gamma and beta : Scale
  - If beta = 0 and Gamma = 1, great job
  - If beta = 10 and gamma = 4, horrendous job

- Do not use FC for converting 2D to 1D

- In CNN, a neuron is a pixel

- Every single neuron which is activated is a pixel

- In FC, we have only 1 channel

- In FC, spatial information is lost

- Any transformation in FC, leads to new vector and doesn't signify previous one

- FC is not Translational invariant

- ![FC_Invariant](.\assets\FC_Invariant.JPG)

- FC is no rotational, skew and scale invariant

- FC uses extremely large no of parameters

- VGG

  - 56x56 is a big image considering image recognition
  - 90% parameters are used by FC and only 10% is used by convolution to predict things.

- Global Average Pooling(GAP)

  - Take sum of all channels and divide by image size to get the average.
  - In case of 7x7 divide by 49 to get the average
  - Multiply with no of channels such as 512

- Best convolution is 1x1

- 1x1 is a FC layer if it is looking at 1D data

- Softmax is the last activation function

- No kernel change due to softmax

- Assignment 1 Review

  - Use MP to compress the image

  - Don't use MP near to last layer

  - Use 1x1 to reduce no of channels

  - GAP vs Non GAP

    - GAP uses less no of parameters compared to Non-GAP

    - In case of 7x7x512 with non-gap, parameters will be huge of 7x7x512x10

      

  - Model remains same (Conv 1-> Conv 2-> Conv 3-> MP -> Conv 4-> Conv 5-> Conv 6-> Conv 7-> Conv 8-> Flatten -> Softmax)

  - Always write # of channels at the end of layer

  - Next session formula for RF in case of MP will be introduced

  - Target is under 20K parameters

  - Second

    - Use MP

  - Third

    - Everything is same except no of kernels
    - Less than 20K parameters. 10890
    - Accuracy is bad, still train acc is 99.33 and score is 98.84(reduced), still not bad
    - 99.33 and 98.84 gives us confidence
      - Diff between 100 and 99.33 : 0.67
      - Add 0.67 to 98.84 : 99.51
      - We are going in the right direction as we crossed our target of 99.4(99.51)
      - How much difference is there b/w 100 and train acc : 100 - 99.33 = 0.67
      - Prefer a model with small gap b/w train and test acc
      - We know we could hit our acc with this model

  - Fourth

    - 3 -> 4. Allowed to change only one thing
    - Introduced BatchNormalization(BN)
    - Add BN after every layer except last layer
    - BN allows increase/decrease amplitude of a feature
    - BN adds loudness 
    - Parameters increased to 11260
    - Adding BN will jump parameters slightly
    - Parameters added will be in two categories
    - BN increases time per epoch
    - Model doesn't look very well
    - Train acc is almost similar 99.39 compared to 99.33
    - However, there is slight increase in val accuracy 98.97 compared to 98.84
    - Ran these multiple times and showing the least score
    - BN is not going to show good output in MNIST
    - The accuracy doesn't improve in case of MNIST with black and white images. However, acc will improve more with color images as in CIFAR10 dataset

  - Fifth

    - Every model, we are going to change only one thing
    - In fourth, kernel values are small
    - Used only 10, 20 these kind of kernels
    -  We are pretty under target parameters of 20K (11260)
    - This time we are going to increase # of parameters
    - We are keeping one thing constant and changing other thing.
    - We changed the kernel to 16, 32 instead of 10,20
    - Parameters increased to 18366 still within target of 20K
    - Added Validation to the model validation_data=(X_test, y_test). This will test model every epcoh.
    - As we added validation to the model,
      - acc is 99.8 with val_acc 99.05
      - Val_acc varies as 98.65, 99.00, 98.95, 99.12, with a max of 99.13
    - If validation takes more time, better not to use it. As in case of large dataset.
    - We should add validation in between in case its time consuming. As in @ 50 epochs as it I knew that accuracy is going to increase after 50 epochs
    - People report highest acc in papers
    - Now train acc : 99.80 and test acc : 99.05. A gap of 0.75 which is high and is not good.
    - The gap is 0.2 (100 -99.80)
    - Add 0.2 to test acc of 99.05, (99.05 + 0.2) = 99.25
    - Its sad to see that adding 0.2 to 99.05 will max make it to 99.25 which is still lesser than targeted one.
    - This is good, as we can see a disease here and that disease is called as overfitting.
    - Overfitting is subjective
    - A person who is fat can be perceived with different views of FAT and Non-FAT
    - Consider a training acc of 100% and Test acc happens to be 80%
      - Do you think above model is ovefitting?
      - State of art for this problem was 74
      - The mode is not overfitting as its closer to world's best model
      - 
    - In our example we had train acc of 99.80 and test acc of 99.05, still we are saying its overfitting.
    - BN and Dropout helps fix overfitting
    - Are you going take crocin today?
    - Why to add something to your network without seeing a disease
    - Model is learning and started to mug up things means overfitting.
    - We need to regularize it
    - We are going to add dropout

  - Sixth

    - Exact same code as in Fifth with a single change of adding a dropout

    - Add small values of dropout at every layer except the last one. 0.25 is a large value. MNIST it may work and not with complex datasets

    - After dropout, 

      - Parameters remains same as 18,366

      - Train (98.34) and Val acc (98.61)gap : 100 - 98.34 = 1.66. Add it to val_acc : 98.61+1.66 = 100.27

      - Dropout is going to reduce the gap b/w train and test acc

      - Sometimes, adding dropout might reduce the acc

      - 99 -> 89 ==== 90 -> 86

      - Looks a decrease of acc from 89 to 86, however its relative as scale has been changed from 99 to 90 which means an improvement in acc

      - There is a scope of further improvement as we just crossed 90.

      - Dropout may reduce the accuracy as it may make your model slightly harder to learn

        ![Dropout and Overfitting](.\assets\Dropout_Overfitting.JPG)

      - Whenever we using a regularization technique, we are trying to reduce the train and test acc gap and doesn't care about the absolute values

        ![Train and test accuracy](.\assets\train-test.JPG)

      - 

    - Seventh

      - Learning rate

- Image displayed for a number in MNIST is in yellow. It's a OpenCV ColorMap

- 

#### Clarifications

- How do we visualize that '1x1 is a feature merger?'
- GAP vs Non-GAP
- More examples of overfitting?
- How do we verify model is learning?
- 
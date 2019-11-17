# EIP4 Assignment 1 - Praveen Raghuvanshi

### Model Test Accuracy - 99.08%

print(score)

 [0.05172299045256509, 0.9908] 



### Definitions

1. **Convolution**

   It is the process of revolving a filter over a portion of an image to generate an output. Mathematically, it's a multiplication of matrices to produce an output matrix. 

2. **Filters/Kernels**

   It comprise of different attributes of an image such as edges, gradients, patterns, parts of objects and objects which gets convolved over different part of image.

3. **Epochs**

   It is one full pass of convolution of kernel over the whole dataset. It comprise of forward and backward pass.

4. **1x1 Convolution**

   It's a special type of convolution used to reduce the filters without changing other parameters.

5. **3x3 Convolution**

   Its a standard convolution with 3x3 kernel and reduces the dimensionality by 2 in every layer.  With 3x3 we can make any size of kernel.

6. **Feature Maps**

    It is the output of feature extraction through a kernel in a convolution process. 

7. **Activation Function**

   It signifies how neurons gets fired/activated using different parameters

8. **Receptive Field**

   Its the area seen by one pixel of a kernel while convolving. A local receptive field scope is the previous layer only and its always size of a kernel. A global receptive field is the area seen by a pixel in last layer.
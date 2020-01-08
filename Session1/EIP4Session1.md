# EIP4 - Session 1 : 11-Nov-2019

#### Video 

 https://www.youtube.com/watch?v=ATDwNCwxSQo 

[![EIP4 Session 1](http://img.youtube.com/vi/ATDwNCwxSQo/0.jpg)](http://www.youtube.com/watch?v=ATDwNCwxSQo)

### Canvas Links

- [Today we can read brains using neural networks now!](https://www.biorxiv.org/content/10.1101/787101v2.full)
- [Video - Drawing timelapse](https://youtu.be/EuBiaRO8QOk)
- 



#### Notes

- Follow [Andrej Karpathy]( https://cs.stanford.edu/people/karpathy/)  and [Yann LeCun]( http://yann.lecun.com/)

- Read Research paper --> Scroll to bottom --> Search for github link --> Look for code and understand from it. No mathematics :-)

- Receptive field of last layer has to be the size of object

- For initial understanding, size of image is size of object

- How do we determine receptive field at every layer?

- Image and channel are used interchangebly
  Image/Channel size : 4x4
  Kernel : 3x3
  Output: 2x2

- Feature map is the output of feature extraction through a kernel

- With 3x3 we can make any size of kernel

- We always have odd numbered kernel such as 3x3, 5x5 etc in order to maintain axis symmetry in order to determine what is left/right

- Every 3x3 is a superset of all the possible 2x2 in the world

- With every 3x3 kernel, we reduce dimensionality by 2. For e.g 5x5 becomes 3x3.... similarly 9x9 becomes 7x7

- Standardization: diving pixel values by 255 in order to have everything between 0 and 1

- Blue channel is input, purple is kernel and Green is output

- Image(3x3) convolved with kernel(3x3) --> 1x1 output

- Adding small kernels helps reduce parameters and increases layers such as below
  Direct : Image(5x5) convolved with kernel(5x5) --> 1x1 output --> 25 parameters
  Layer 1: Image(5x5) with kernel(3x3) --> 3x3 output --> 9
  Layer 2: Image(3x3) with kernel(3x3) --> 1x1 output --> 9
  Total Parameters: 9 + 9 = 18

- Convolving with Kernel of 3x3 uses 9 parameters
  Total Parameters if 3x3 is in all layers = 9 * num_layers

  For e.g Image(11x11) : 121 
  11 -> 9 -> 7 -> 5 -> 3 -> 1 : Total Parmeters : 9 * 5[11,9,7,5,3] = 45
  Significant reduction in parameters

- **Local Receptive Field :** Area seen by one pixel of a kernel while convolving.

  In case of below LRF of highlighted pixel in Red in middle layer is 9(3x3 in left layer)

  ![Local Receptive Field](.\assets\Local_Receptive_Field.JPG)

  LRF doesn't know about other areas apart from area being covered by a kernel.

  LRF is always size of kernel

- Global Receptive Field: Area seen by a pixel in last layer(Rightmost layer)

  In case of below, GRF of green pixel is 25(5x5)

  ![](.\assets\Global_Receptive_Field.JPG)

  Aim is to have GRF equal to size of image. It's not always true.

  With addition of layer with a kernel of 3x3, GRF is increased by 2.

  The layer which is going to make prediction(generally last layer) should have GRF near to image size

  For e.g if we have an image of 100 x 100 and there are three layers (3, 5, 7), last layer can see  7x7 and doesn't know anything about other pixels. So in this case, last layer should have GRF near to 100. It could be 100, 105 etc.

  We need as many layers as to reach the global receptive field.

  For 400x400 image, we need 200 layers which is very high compared to different models we have such as Resnet50, 121 etc.

  ![LRF vs GRF](.\assets\LRF_GRF_400.JPG)

  Trying to make 399x399 kernel

- **Max Pooling**

  ![Max Pooling](.\assets\MAX_POOLING.jpg)

  Loss of features in small image. How much did we miss? 

  Some features are screaming loud and we didn't miss them.

  Max Pooling reduces the dimension by half in case we use 2x2 MP

  ![Max Pooling pyramid](.\assets\MAX_POOLING_Pyramid.jpg)

  ![MaxPooling Maths](.\assets\MAX_POOLING_Maths.jpg)

- Each kernel is a feature extractor

- No of channels in input has not relationship with no of channels in output

- For an image of 400x400 with a receptive field of 11x11 at layer 5, we'll not be able to see edges at below 11x11

- MP layer varies with size of image. 

- Why MP after 5 layers?

  - For large image such as 400x400, we are going till layer 5
  - For smaller images such as MNIST 28x28, we can go with MP after couple of layers

- Imagenet has 1000's of classes and MP layer defines how capable the network is.

- Doesn't matter how good are my layers before MP, If my network is unable to extract information after MP - 5 layers, our network is not going to work.

- ![Max Pooling Maths -1](.\assets\MAX_POOLING_Maths_1.jpg)

-  Cat vs Dog problem:

  - Network should learn different types of cats and dogs
  - Network should learn different background in order to remove it
  - DNN objective is not only to detect the features(cat, dog) but also filter them(background)
  - The most difficult task in life is to say NO and to say No, we need to know for what we need not to say NO.

- 3x3 is a feature extractor

- 1x1 is going to combine those features which are contextually linked together

- 1x1 acts like a DJ to combine songs and not create a song

- Through 1x1, we can create 12 colors from RGB and vice versa

- 1x1 reduces no of channels

- 1x1 is a z-dimension reduction

- MP is x and y dimension reduction

- 32x32x10 being convolved by 1x1x10x4

- For this course 1x1 is going to be used to reduce no of channels

- Conversion to float as we have infinite no of images between 0-1.

-  

#### Clarifications

- What is a receptive field? How do we calculate it and what is the significance of it?
- Significance of 1x1? Reduce no of channels keeping parameters same ?
- What is Z-Channel Dimension Reductionality?
- Gif of '3x3! How convolutions actually work'?
- How we relate concept of receptive field in code?
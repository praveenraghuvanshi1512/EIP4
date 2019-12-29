# EIP4 Assignment 5 - Praveen Raghuvanshi

## Final best validation accuracy: --%

- Best Val Accuracy(46th Epoch): **89.29%**
- Total parameters: **274,442**
- No of Epochs: **50**
- ipynb file : [Solution](EIP4_A5_Praveen_Raghuvanshi.ipynb)

```python
Epoch 43/50
Learning rate:  1e-05
 - 49s - loss: 0.3309 - acc: 0.9386 - val_loss: 0.4771 - val_acc: 0.8929

```

##### Model 

- Used V1 from https://keras.io/examples/cifar10_resnet/
- Changes
  - Learning Rate: Epcoh/lr : 1-20(0.001) --> 21-35(0.0001) --> 36->45(1e-05) --> 46-50(1e-06)

```python
lr = 1e-3

if epoch > 45:
    lr *= 1e-3
    elif epoch > 35:
        lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
print('Learning rate: ', lr)
```

##### Logs

```python

```



#### Analysis

##### Pre-Analysis

- Dataset : Custom image dataset

- Default Model: VGG16

- Image size : 224 x 224 x 3

- Total Images: 13573

  - Train data: 11537
  - Test data: 2036

- Classes (8) : [gender, imagequality, age, weight, carrybag, footwear, emotion, bodypose]

- Code Fixes

  - valid_gen: Replace train_df with val_df
  - neck = Dropout(0.3)(in_layer) : Replaced in_layer with neck
  
  

| S.No | Iteration       | Architecture           | Total Parameters | Trainable Parameters | Freeze layers | Batch Size         | Epochs  | Val Acc (Best) Gender | Time(Per epoch(secs)) | Model Changes                                                | Remark                                                       |
| ---- | --------------- | ---------------------- | ---------------- | -------------------- | ------------- | ------------------ | ------- | --------------------- | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1    | Base            | VGG1, Weights = 'None' | 28,089,051       | 13,374,363           | VGG16 default | Train: 32, Val: 64 | 10      | 62.48                 | 65s                   | batch size - 256                                             | N/A                                                          |
| 2    | Solution(Final) |                        |                  |                      |               |                    |         |                       |                       |                                                              | Best acccuracy                                               |
| 3    | First           | VGG1, Weights = 'None' | 28,089,051       | 13,374,363           | VGG16 default | Train: 32, Val: 64 | 10      | 57.81                 | 43                    | Code fixes                                                   | Acc reduced.                                                 |
| 4    | Second          | VGG1, Weights = 'None' | 28,089,051       | 13,374,363           | VGG16 default | Train: 32, Val: 64 | 50      | 67.19                 | 50                    | Code Fixes + Epochs : 50                                     | Acc improved.                                                |
| 5    | Third           | ResNet20v1             | 274,442          |                      |               | 256                | 50      | 57.76                 | 46s                   | Second + Checkpoint + (lre-1 -> lre-2) + epoch(30 -> 20)     | Acc reduced.                                                 |
| 6    | Fourth          | ResNet20v1             | 274,442          |                      |               | 256                | 10 + 30 | 64.92                 | 75                    | (lre-1 -> lre-4) + epoch(10 -> 40)                           | Acc improved but less than best. Plot converging             |
| 7    | Fifth           | ResNet20v1             | 274,442          |                      |               | 256                |         | 63.76                 | 88s                   | (lre-1 -> lre-4 -> lre-6) + epoch(10 -> 20 -> 30)            | Acc reduced. Plt divereged a lot                             |
| 8    | Sixth           | ResNet20v1             | 274,442          |                      |               | 64                 |         | 65.78           | 88s                    | (lre-1 -> lre-2 -> lre-3) + Epoch(20->35->45)            | Acc improved                                                  |
| 9    | Seventh         | ResNet20v1             | 274,442          |                      |               | 32                 |         | 66.89           | 42s                  | (lre-1 -> lre-2 -> lre-3->lre-4->lre-3) + Epoch(7->13->28->47) | Acc improved, Plot converged better                          |
| 10   | Eighth          | ResNet20v1             | 274,442          |                      |               | 32                 |         | 81.85                 | 55                    | lr=0.0001, checkpoint, save and load weights | Acc dropped                                                  |
| 11   | Ninth           | ResNet20v1             | 274,442          |                      |               | 32                 |         | 81.97                 | 70                    | lr reset to 0.001, apply cutout(5)                           | Acc reduced                                                  |
| 12   | Tenth           | ResNet20v1             | 274,442          |                      |               | 16                 |         | 84.98                 | 88                    | Removed Cutout, Reduced batch size 32->16                    | Buffered data was truncated after reaching the output size limit. Acc redcued, time increased, |
| 13   | Eleventh        | ResNet20v1             | 274,442          |                      |               | 128                |         | 88.09                 | 27                    | LR(25->35->45) : 0.001 -> 0.0001 -> 0.00001, BS 128, verbose=2 | Acc improved and crossed threshold of 88%.                   |
| 14   | Thirteenth      | ResNet20v1             | 274,442          |                      |               | 32                 |         | 89.20                 | 50                    | bs 32                                                        |                                                              |
|      |                 |                        |                  |                      |               |                    |         |                       |                       |                                                              |                                                              |
|      |                 |                        |                  |                      |               |                    |         |                       |                       |                                                              |                                                              |
|      |                 |                        |                  |                      |               |                    |         |                       |                       |                                                              |                                                              |

## Learning Rate 

- VGG16

![](D:\Praveen\SourceControl\github\praveenraghuvanshi1512\EIP4\Session5\assets\learning_rate_labelled.JPG)

![Loss curve](..\assets\loss_curve.JPG)

- As per the above curve, loss is stagnant till 10e-2 and after that it shoots up exonentialy
- Range: 10e-10 to 10e+1
- Max LR: 10e-2

### Random tips

- [Difference between Loss, Accuracy, Validation loss, Validation accuracy in Keras](https://www.javacodemonk.com/difference-between-loss-accuracy-validation-loss-validation-accuracy-in-keras-ff358faa)

  - But with **val_loss**(keras validation loss) and **val_acc**(keras validation accuracy), many cases can be possible like below:

  1. *val_loss* starts increasing, *val_acc* starts decreasing. This means model is cramming values not learning
  2. *val_loss* starts increasing, *val_acc* also increases.This could be case of overfitting or diverse probability values in cases where softmax is being used in output layer
  3. *val_loss* starts decreasing, *val_acc* starts increasing. This is also fine as that means model built is learning and working fine.

## Baseline

- {'**age_output_acc': 0.2525173611111111,** 'age_output_loss': 12.047996536890667, 'bag_output_acc': 0.5634548611111111, 'bag_output_loss': 7.0362762557135685, 'emotion_output_acc': 0.7079861111111111, 'emotion_output_loss': 4.706707759698232, 'footwear_output_acc': 0.3684895833333333, 'footwear_output_loss': 10.17874510553148, '**gender_output_acc': 0.5630208333333333**, 'gender_output_loss': 7.043271933661567, 'image_quality_output_acc': 0.5561631944444444, 'image_quality_output_loss': 7.15380405055152, 'loss': 60.13644635942247, 'pose_output_acc': 0.6186631944444444, 'pose_output_loss': 6.146423045794169, 'weight_output_acc': 0.6387152777777778, 'weight_output_loss': 5.823221646414863}

## First

- 

## Second

- {**'age_output_acc': 0.40423387096774194**, 'age_output_loss': 1.3888073005983907, 'bag_output_acc': 0.5761088709677419, 'bag_output_loss': 0.901054116987413, 'emotion_output_acc': 0.7192540322580645, 'emotion_output_loss': 0.8880955519214753, 'footwear_output_acc': 0.5922379032258065, 'footwear_output_loss': 0.9058221309415756, '**gender_output_acc**': **0.6577620967741935**, 'gender_output_loss': 0.619186439821797, 'image_quality_output_acc': 0.5413306451612904, 'image_quality_output_loss': 0.9853553483563084, 'loss': 7.576153001477642, 'pose_output_acc': 0.6023185483870968, 'pose_output_loss': 0.9383813008185355, 'weight_output_acc': 0.6512096774193549, 'weight_output_loss': 0.9494507774229972}

## Third

- {**'age_output_acc': 0.39818548387096775**, 'age_output_loss': 1.4288250438628658, 'bag_output_acc': 0.5645161290322581, 'bag_output_loss': 0.9182965717008037, 'emotion_output_acc': 0.7051411290322581, 'emotion_output_loss': 0.924340630731275, 'footwear_output_acc': 0.5725806451612904, 'footwear_output_loss': 0.9440910239373485, '**gender_output_acc**': **0.5504032258064516**, 'gender_output_loss': 0.6858665308644695, 'image_quality_output_acc': 0.5655241935483871, 'image_quality_output_loss': 0.9652477368231742, 'loss': 7.781947505089544, 'pose_output_acc': 0.6118951612903226, 'pose_output_loss': 0.9358331945634657, 'weight_output_acc': 0.6376008064516129, 'weight_output_loss': 0.9794468014470993}

## Fourth

- {'**age_output_acc': 0.40473790322580644**, 'age_output_loss': 1.3867902217372772, 'bag_output_acc': 0.5549395161290323, 'bag_output_loss': 0.9087540411180065, 'emotion_output_acc': 0.7253024193548387, 'emotion_output_loss': 0.8735187534363039, 'footwear_output_acc': 0.5987903225806451, 'footwear_output_loss': 0.8741694265796293, '**gender_output_acc**': **0.6431451612903226**, 'gender_output_loss': 0.6236961060954679, 'image_quality_output_acc': 0.5549395161290323, 'image_quality_output_loss': 0.9643754286150779, 'loss': 7.525375289301718, 'pose_output_acc': 0.5992943548387096, 'pose_output_loss': 0.9322860048663232, 'weight_output_acc': 0.639616935483871, 'weight_output_loss': 0.9617853933765043}

## Fifth

- {**'age_output_acc': 0.3911290322580645,** 'age_output_loss': 1.3899336899480512, 'bag_output_acc': 0.5700604838709677, 'bag_output_loss': 0.9080106731384031, 'emotion_output_acc': 0.7081653225806451, 'emotion_output_loss': 0.9157512168730458, 'footwear_output_acc': 0.6174395161290323, 'footwear_output_loss': 0.8462165613328257, **'gender_output_acc': 0.6376008064516129**, 'gender_output_loss': 0.6264831692941727, 'image_quality_output_acc': 0.561491935483871, 'image_quality_output_loss': 0.9577924936048446, 'loss': 7.50841397623862, 'pose_output_acc': 0.6063508064516129, 'pose_output_loss': 0.9069510602181957, 'weight_output_acc': 0.6370967741935484, 'weight_output_loss': 0.957275019538018}

## Sixth

- {**'age_output_acc': 0.3996975806451613**, 'age_output_loss': 1.3903789904809767, 'bag_output_acc': 0.5509072580645161, 'bag_output_loss': 0.9225210374401461, 'emotion_output_acc': 0.7172379032258065, 'emotion_output_loss': 0.8874025287166718, 'footwear_output_acc': 0.5922379032258065, 'footwear_output_loss': 0.8896488258915562, **'gender_output_acc': 0.6557459677419355,** 'gender_output_loss': 0.625312082229122, 'image_quality_output_acc': 0.5408266129032258, 'image_quality_output_loss': 0.9790309456086927, 'loss': 7.620441021457795, 'pose_output_acc': 0.623991935483871, 'pose_output_loss': 0.9077457208787242, 'weight_output_acc': 0.6184475806451613, 'weight_output_loss': 1.018400865216409}

## Seventh

- {'**age_output_acc': 0.3956653225806452**, 'age_output_loss': 1.3899654919101345, 'bag_output_acc': 0.5650201612903226, 'bag_output_loss': 0.8993718816388038, 'emotion_output_acc': 0.7172379032258065, 'emotion_output_loss': 0.8874586762920502, 'footwear_output_acc': 0.6003024193548387, 'footwear_output_loss': 0.8714978598779247, **'gender_output_acc': 0.6663306451612904**, 'gender_output_loss': 0.6118250643053362, 'image_quality_output_acc': 0.5700604838709677, 'image_quality_output_loss': 0.948535273152013, 'loss': 7.453802508692587, 'pose_output_acc': 0.6174395161290323, 'pose_output_loss': 0.8946676350408985, 'weight_output_acc': 0.6461693548387096, 'weight_output_loss': 0.950480584175356}

## Eighth

- First Pass

  {**'age_output_acc': 0.4168346774193548**, 'age_output_loss': 1.3811535681447675, 'bag_output_acc': 0.594758064516129, 'bag_output_loss': 0.8810908986676124, 'emotion_output_acc': 0.7207661290322581, 'emotion_output_loss': 0.8915172257731038, 'footwear_output_acc': 0.6108870967741935, 'footwear_output_loss': 0.8420301060522756, **'gender_output_acc': 0.704133064516129**, 'gender_output_loss': 0.5656912538313097, 'image_quality_output_acc': 0.5504032258064516, 'image_quality_output_loss': 0.9514134564707356, 'loss': 7.310308287220616, 'pose_output_acc': 0.6270161290322581, 'pose_output_loss': 0.8348056885503954, 'weight_output_acc': 0.639616935483871, 'weight_output_loss': 0.9626061685623661}

- Second Pass

  {**'age_output_acc': 0.4112903225806452,** 'age_output_loss': 1.3812514351260277, 'bag_output_acc': 0.5871975806451613, 'bag_output_loss': 0.8949945376765344, 'emotion_output_acc': 0.7207661290322581, 'emotion_output_loss': 0.8791487024676415, 'footwear_output_acc': 0.6199596774193549, 'footwear_output_loss': 0.8413058711636451, **'gender_output_acc': 0.7167338709677419**, 'gender_output_loss': 0.5638810088557582, 'image_quality_output_acc': 0.5448588709677419, 'image_quality_output_loss': 0.9433998119446539, 'loss': 7.254877275036227, 'pose_output_acc': 0.6517137096774194, 'pose_output_loss': 0.7863317593451469, 'weight_output_acc': 0.639616935483871, 'weight_output_loss': 0.9645640965430967}

- Third Pass

  {**'age_output_acc': 0.4117943548387097,** 'age_output_loss': 1.3759757049622074, 'bag_output_acc': 0.5952620967741935, 'bag_output_loss': 0.8772198288671432, 'emotion_output_acc': 0.7207661290322581, 'emotion_output_loss': 0.8781320279644381, 'footwear_output_acc': 0.6194556451612904, 'footwear_output_loss': 0.829098726472547, **'gender_output_acc': 0.7247983870967742,** 'gender_output_loss': 0.5370296585944391, 'image_quality_output_acc': 0.5428427419354839, 'image_quality_output_loss': 0.941912337656944, 'loss': 7.160585418824227, 'pose_output_acc': 0.6668346774193549, 'pose_output_loss': 0.7611693432254176, 'weight_output_acc': 0.6401209677419355, 'weight_output_loss': 0.9600477468582892}

## Ninth

- Do not freeze layers on first iteration

- remove relu in build_tower from last layer

- Use losses commented code for different class

- Add batch normalization

- random_state - 42

- First pass

  - ```python
    31/31 [==============================] - 18s 568ms/step
    {'age_output_acc': 0.4112903225806452,
     'age_output_loss': 1.4392743341384395,
     'bag_output_acc': 0.5887096774193549,
     'bag_output_loss': 0.9180686435391826,
     'emotion_output_acc': 0.7434475806451613,
     'emotion_output_loss': 0.8437503441687553,
     'footwear_output_acc': 0.5655241935483871,
     'footwear_output_loss': 0.9463730646717933,
     'gender_output_acc': 0.578125,
     'gender_output_loss': 0.676357961470081,
     'image_quality_output_acc': 0.5982862903225806,
     'image_quality_output_loss': 0.9898216378304266,
     'loss': 7.746357717821675,
     'pose_output_acc': 0.6199596774193549,
     'pose_output_loss': 0.9625091379688632,
     'weight_output_acc': 0.6441532258064516,
     'weight_output_loss': 0.9702025863432115}
    ```

  - model.007.h5

## Tenth

- **DenseNet121**
- First Pass:
  - Gender acc improved from 57.81 to 79.83
  - Age acc reduced from 41.11 to 36.54
  - **{'age_output_acc': 0.3654233870967742,** 'age_output_loss': 1.6884689292600077, 'bag_output_acc': 0.6139112903225806, 'bag_output_loss': 1.1695313376765097, 'emotion_output_acc': 0.6648185483870968, 'emotion_output_loss': 0.9338526668087128, 'footwear_output_acc': 0.625, 'footwear_output_loss': 1.0108790609144396**, 'gender_output_acc': 0.7983870967741935,** 'gender_output_loss': 0.5861075337856047, 'image_quality_output_acc': 0.5393145161290323, 'image_quality_output_loss': 1.0961351644608281, 'loss': 8.803273754735146, 'pose_output_acc': 0.6648185483870968, 'pose_output_loss': 1.2090687155723572, 'weight_output_acc': 0.6386088709677419, 'weight_output_loss': 1.1092303818272007}
- Second Pas:
  - lr - 0.0001
  - {**'age_output_acc': 0.37651209677419356**, 'age_output_loss': 1.954458225157953, 'bag_output_acc': 0.5877016129032258, 'bag_output_loss': 1.369484955264676, 'emotion_output_acc': 0.5846774193548387, 'emotion_output_loss': 1.0346313734208383, 'footwear_output_acc': 0.6204637096774194, 'footwear_output_loss': 1.2012831645627176, '**gender_output_acc': 0.7636088709677419**, 'gender_output_loss': 0.8062966725518627, 'image_quality_output_acc': 0.5126008064516129, 'image_quality_output_loss': 1.34416062601151, 'loss': 10.104683752982847, 'pose_output_acc': 0.7217741935483871, 'pose_output_loss': 1.091904607511336, 'weight_output_acc': 0.5902217741935484, 'weight_output_loss': 1.3024640179449511}

## Eleventh

- Custom
- lr - 0.001
- Earlystopping - patience=10
- First Pass:
  - val_loss : 8.71160
  - model.017.h5
  - {**'age_output_acc': 0.4117943548387097**, 'age_output_loss': 1.4446294230799521, 'bag_output_acc': 0.6209677419354839, 'bag_output_loss': 0.8412172102159069, 'emotion_output_acc': 0.7424395161290323, 'emotion_output_loss': 0.8264846628712069, 'footwear_output_acc': 0.6194556451612904, 'footwear_output_loss': 0.8301186638493692, '**gender_output_acc': 0.813508064516129**, 'gender_output_loss': 0.5340268208134559, 'image_quality_output_acc': 0.4753024193548387, 'image_quality_output_loss': 1.0578230196429836, 'loss': 9.261353277391002, 'pose_output_acc': 0.7323588709677419, 'pose_output_loss': 0.6705555050603805, 'weight_output_acc': 0.6441532258064516, 'weight_output_loss': 0.9316127569444718}
- Second Pass:
  - lr = 0.0001
  - load 017 and re-train



### References

- [Overfitting](https://stats.stackexchange.com/questions/128616/whats-a-real-world-example-of-overfitting/128625#128625)

![Overfitting](..\assets\overfitting)
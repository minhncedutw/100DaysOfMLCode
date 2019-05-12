# 100DaysOfMLCode

#### --------------------------------------------------
## Day 1(2019May03): Practice YOLO3 keras

**Practice source code:** https://github.com/minhncedutw/prac-keras-yolo3/blob/master/train_re.py

#### --------------------------------------------------
## Day 2(2019May04): Practice SSD keras

**Practice source code:** https://github.com/minhncedutw/prac-keras-ssd/blob/master/ssd300_training.py

#### --------------------------------------------------
## Day 3(2019May05): Practice YOLO2 keras - Design pattern of project

**Some best practices:**
 - refactored backend
 - neat & concise learner
 - neat configuration from `config.json`
 - import absolute path
 

**^^Design pattern of project:**
 - import absolute path
 - [refactor backend](https://github.com/minhncedutw/handbook/blob/master/python_tips/backend.py)

**Practice source code:** https://github.com/minhncedutw/prac-keras-yolo2/blob/master/train_yolo2_re.py

**Handling imbalance data** https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28

#### --------------------------------------------------
## Day 4(2019May06): Practice Unet fastai(lesson3-camvid)

**Practice source code:** https://github.com/minhncedutw/prac-fastaiv1-dl1/blob/master/prac/prac_lesson3-camvid_2.ipynb

**Fastai training pipeline**
 - prepare:
    + data-bunch
    + architecture of model
    + learner
 - training:
    + train small data-bunch(small images, often ~1/2 image size):
        * freeze head layers. find learning rate. train with slice of lr1 = minimum_peak /10
        * unfreeze head layers. train with slice of lrs = [lr1/400, lr1/4] 
    + train big data-bunch(full size images):
        * freeze head layers. find learning rate. train with slice of lr2 = minimum_peak /10
        * unfreeze head layers. train with slice of lrs = [..., lr2/10] 

#### --------------------------------------------------
## Day 5(2019May07): Practice PointNet keras - keras Lambda and keras Custom Layer

**^^Lambda & Custom Layer** [practice](https://github.com/minhncedutw/handbook/blob/master/python_tips/keras_tips.md)

**Practice source code:** [PointNet model](./prac_codes/day05/)

**Question?** 
 - is `keras.layers.Dot` equal `Lamda(tf.matmul)`?
```python
net_transformed = Dot(axes=(2, 1))([net, ftransform])
# [Source](https://github.com/HPInc/pointnet-keras/blob/master/model.py)
```

#### --------------------------------------------------
## Day 6(2019May08): Practice one hot encoding/decoding

**Practice source code:** [commit](https://github.com/minhncedutw/handbook/commit/e0ddf8848c9ab3612c4547263c0cd4ec20ff89f7)

#### --------------------------------------------------
## Day 7(2019May09): Multithread(run a function in background)

**Practice source code** [commit](https://github.com/minhncedutw/handbook/commit/e2ecac58ac5447ae51c0f18a8ec0b6cf10a0508c)

#### --------------------------------------------------
## Day 8(2019May10): Build my own utilities lib(python, keras, pointcloud)

**Practice source code** [commit](https://github.com/minhncedutw/handbook/commit/ff54ab5400d52de3e15a43fbe8c01a62b751f2f0)

#### --------------------------------------------------
## Day 9(2019May11): Practice Resnet50 fastai(lesson3-planet)

**Brief train process:**
 - Train small data(image size is a half)
    + train big lr: lr1 = minimum peak / 10
    + train small slice of lr: lr2 = minimum peak / 10 -> lr1 / 5
 - Train big data
    + train big slice of lr: lr1 = minimum peak / 10
    + train small slice of lr: lr2 = minimum peak / 10 -> lr1 / 5
    
**Notice**
 - above process can loop with different size of images, but not sure whether it is able to improve
 - can select divide by 5, 6, ...10

**Practice source code** [source](./prac_codes/day09/)

#### --------------------------------------------------
## Day 10(2019May12): Focal loss

**Cross-entropy**
 - binary cross-entropy
 - categorical cross-entropy
 - weighted categorical cross-entropy
 - focal:
    + binary focal cross-entropy
    + categorical focal cross-entropy
    
**Difference between `categorical_crossentropy` and `sparse_categorical_crossentropy`:**
 - categorical_crossentropy: target is one-hot encoded
 - sparse_categorical_crossentropy: target is integers
Source: https://jovianlin.io/cat-crossentropy-vs-sparse-cat-crossentropy/
    
Good article: https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4

**Practice source code** [commit](https://github.com/minhncedutw/handbook/commit/fcd84c38acec1ffba043cf93bf4e0ead8da7a139)

#### --------------------------------------------------
## Day 11(2019May13): PointNet loss

**PointNet Loss** pass layer of network as argument to loss function:
 - name a fixed name for the layer
 - get the layer from model by name: model.get_layer(...)
```python
model.compile(loss=[pointnet_loss(xtran=model.get_layer(name='xtran2'), reg_weight=0.001)], metrics=["accuracy"], optimizer=adam)
``` 

**l2_loss**
 - l2_loss= sum(t ** 2) / 2
![](https://raw.githubusercontent.com/ritchieng/machine-learning-nanodegree/master/deep_learning/deep_neural_nets/dnn12.png)

**Practice source code** [source](./prac_codes/day11)

#### --------------------------------------------------
## Day 12(2019May14): neat Learner template based on YOLO2 project

**Practice source code** 
 - [Learner template](./prac_codes/day12/LearnerTemplate.py)
 - [bonus FeatureExtractor template](./prac_codes/day12/BackendTemplate.py)


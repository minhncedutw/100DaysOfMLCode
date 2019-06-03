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

**Practice source code:** [PointNet model](old2/day05/)

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

**Practice source code** [source](old2/day09/)

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

**Practice source code** [source](old2/day11)

#### --------------------------------------------------
## Day 12(2019May14): neat Learner template based on YOLO2 project

**Practice source code** 
 - [Learner template](old2/day12/LearnerTemplate.py)
 - [bonus FeatureExtractor template](old2/day12/BackendTemplate.py)

#### --------------------------------------------------
## Day 13(2019May15): Run Learning rate finder - keras

**Learning rate problems**
 - Learning rate finding
 - Learning rate annealing
Source: [SETTING THE LEARNING RATE OF YOUR NEURAL NETWORK](https://www.jeremyjordan.me/nn-learning-rate/)

**Practice source code** [use lr_finder on cifar10+mnist training](old2/day13)

#### --------------------------------------------------
## Day 14(2019May16): Code learning rate finder - keras

**Practice source code** [Code lr_finder](old2/day14)

#### --------------------------------------------------
## Day 15(2019May17): Run one-cycle - keras

**Practice source code** [run plot_clr](https://github.com/minhncedutw/prac-keras-one-cycle-lr.git)

**Usage**
```python
from clr import OneCycleLR
lr_manager = OneCycleLR(max_lr=0.02, maximum_momentum=0.9, verbose=True)
```

#### --------------------------------------------------
## Day 16(2019May18): Cyclical Learning Rate (CLR) - keras



#### --------------------------------------------------
## Day 16(2019May18): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May19): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May20): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May21): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May22): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May23): Stochastic Gradient Descent with Warm Restarts (SGDR) - keras

#### --------------------------------------------------
## Day 16(2019May24): Publish Machine Learning API with Flash

```python
import json
import numpy as np
from flask import Flask, request, jsonify
# ────────────────────────────────────────────── I ──────────
#   :::::: A P I : :  :   :    :     :        :          :
# ────────────────────────────────────────────────────────
app = Flask(__name__)
@app.route('/api/', methods=['POST'])
def make_prediction():
    data = request.get_json()
    points = np.array(data['points'])

    labels = learner.predict(x=points, verbose=1)

    return jsonify(labels.tolist())

if __name__ == '__main__':
    # ──────────────────────────────────────────────────────────────────────────────────── I ──────────
    #   :::::: L O A D   C O N F I G   P A R A M E T E R S : :  :   :    :     :        :          :
    # ──────────────────────────────────────────────────────────────────────────────────────────────
    learner = create_learner(input_shape=input_shape, labels=lbl_names)
    learner.load_weight(weight_path=path)
    
    app.run(debug=False, host='127.0.0.1', port=5000)
```

#### --------------------------------------------------
## Day 16(2019May25): Safe thread for Keras(Tensorflow) in multi-thread apps

When **load weight/model**, should **save graph**:
```python
def load_weight(self, weight_path: Union[str, Path]):
    success = self.model.load_weights(weight_path)
    self.model._make_predict_function() # safe thread
    self.graph = tf.get_default_graph() # save graph of current model
    self.predict(np.ones(self.input_shape)) # initialize the network in the 1st time
    print('Load weight {}: {}'.format(weight_path, success))
```
When **do prediction**, should **load the graph** saved before:
```python
    def predict(self, x, verbose=0):
        # Normalize data before feed to PointNet
        # x = self.normalizer(x)

        # Predict labels
        with self.graph.as_default():
            prob = self.model.predict(x=np.expand_dims(a=x, axis=0))

        # Decode probabilities to labels
        prob = np.squeeze(prob)  # remove dimension of batch
        pred_labels = prob.argmax(axis=1)

        return pred_labels
```
>futhermore, at the end of loading weight, should do foo prediction to warmup the network model.

#### --------------------------------------------------
## Day 17(2019May26): DataBunch(manage train-valid generators from directory, bs, ...)

**Principle**
 - Generator loads data from file paths(which are distributed into train paths and valid paths)
 - DataBunch receives directory and parameters, then explores data paths and distributes them into train paths and valid paths
 - Created generators are used for learner.
 
**Practice source code** [Code databunch](old2/day17)

#### --------------------------------------------------
## Day 18(2019May27): CppRestSDK(C++ microservice)

**Install CppRestSDK**
 - Firstly, install VSCommunity C++ 2017
 - Secondly, install vcpkg
 - Thirdly, from vcpkg install cpprestsdk

**Practice source code** [Code server-client rest](old2/day18)
 - server listens request: server.cpp(must be run as admin)
 - client send request: client.cpp

**Reference Source** [Revisited: Full-fledged client-server example with C++ REST SDK 2.10](https://mariusbancila.ro/blog/2017/11/19/revisited-full-fledged-client-server-example-with-c-rest-sdk-2-10/)

#### --------------------------------------------------
## Day 19(2019May28): CppRestSDK interact Python Flask

**Practice source code** [Code server-client rest](old2/day19)
 - server listens request: server.cpp(must be run as admin)
 - client send request: client.py (different from client.cpp, client.py has no *delete method*)


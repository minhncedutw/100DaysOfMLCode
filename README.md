# 100DaysOfMLCode

## Day 1(2019May03): Practice YOLO3 keras

**Practice source code:** https://github.com/minhncedutw/prac-keras-yolo3/blob/master/train_re.py


## Day 2(2019May04): Practice SSD keras

**Practice source code:** https://github.com/minhncedutw/prac-keras-ssd/blob/master/ssd300_training.py

## Day 3(2019May05): Practice YOLO2 keras - Design pattern of project

**^^Design pattern of project:**
 - import absolute path
 - [refactor backend](https://github.com/minhncedutw/handbook/blob/master/python_tips/backend.py)

**Practice source code:** https://github.com/minhncedutw/prac-keras-yolo2/blob/master/train_yolo2_re.py

**Handling imbalance data** https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28

## Day 4(2019May06): Practice Unet fastai

**Practice source code:** https://github.com/minhncedutw/prac-fastaiv1-dl1/blob/master/prac/prac_lesson3-camvid_2.ipynb

## Day 5(2019May07): Practice PointNet keras - keras Lambda and keras Custom Layer

**^^Lambda & Custom Layer** [practice](https://github.com/minhncedutw/handbook/blob/master/python_tips/keras_tips.md)

**Practice source code:** [PointNet model](.day5/pointnet_model.py)

**Question?** 
 - is `keras.layers.Dot` equal `Lamda(tf.matmul)`?
```python
net_transformed = Dot(axes=(2, 1))([net, ftransform])
# [Source](https://github.com/HPInc/pointnet-keras/blob/master/model.py)
```
---
layout: post
title: Logistic Regression
date: 2018-08-23
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: sklearn/logistic_regression/logistic_regressionoutput_example.png # Add image post (optional)
tags: [ML, LR] # add tag
---
# Logistic Regression

> 最近决定开始看 sklearn 的源码，万千思绪不知如何下手，也没找到别人看源码的思路，过程分享出来，所以决定自己先一点一点摸索出来吧。现在的想法是先整理出一个关键的脉络，然后将核心的代码单独提取出来，形成只含有每个独立模型的文件。

## 1 从类的定义开始 

不知道以后 sklearn 库会如何变化，为了保持独立性，fork 到我自己仓库中，对应的为我仓库中的文件。

[class LogisticRegression(BaseEstimator, LinearClassifierMixin, SparseCoefMixin)](https://github.com/calfchen/scikit-learn/blob/master/sklearn/linear_model/logistic.py#L965)

从形式上看，LogisticRegression 继承了三个类：

- BaseEstimator：这个类相当的关键，具体关键到什么程度呢，关键到我要另起一篇文章来单独的介绍。[BaseEstimator]()
- LinearClassifierMixin：这个类就是实现
- SparseCoefMixin

![Macbook]({{site.baseurl}}/assets/img/sklearn/logistic_regression/logistic_regressionoutput_example.png)
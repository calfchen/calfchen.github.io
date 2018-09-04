---
layout: post
title: Sklearn Design
date: 2018-08-22
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: # Add image post (optional)
tags: [ML, Estimator] # add tag
---
# Sklearn Design

> 讲到 Estimator，需要从 sklearn 的 API 设计讲起，API 设计的思想可以参考这篇文章 [API design for machine learning software:experiences from the scikit-learn project](https://dtai.cs.kuleuven.be/events/lml2013/papers/lml2013_api_sklearn.pdf)。同时，在这篇博客的结尾，也有文章对应在该项目下的地址。

文章重点梳理：

## 1 Introduction

sklearn library 基于 Numpy 和 Scipy libraries。

## 2 Core API

sklearn 使用基本的三大接口：① <font color="#dd0000">estimator</font>：building and fitting models；② <font color="#dd0000">predictor</font>：makeing predictions；③ <font color="#dd0000">transformer</font>：converting data。

### 2.1 __General principles__

- Consistency:
- Inspection:
- Non-proliferation of classes:
- Composition:
- Sensible defaults:

### 2.2 __Data representation__

matrix representation: numpy 的 narrays 和 scipy 

datasets are encoded as NumPy multidimensional arrays for dense data and as SciPy sparse matrices for sparse data. 

### 2.3 __<font color="#dd0000">Estimator</font>__

The estimator interface is at the core of the library.

- estimator 定义了对象的实例化方法，并实现了 fit 方法来训练数据
- All supervised and unsupervised learning algorithms 都提供了 estimator 接口，所有的 Machine learning tasks like feature extraction, feature selection or dimensionality reduction 也提供了 estimator 接口。
- estimator 的 initialization 和实际的 learning 过程是分开的，estimator 的 initialization 主要需要传入一些超参数(例如 SVM 的松弛因子C)；learning 过程主要通过 fit 的方法来实现，此过程主要通过 data 来学习一些 paramaters(例如线性模型的 coef_)。

To illustrate the initialize-fit sequence：

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(penalty=”l1”)
    clf.fit(X_train, y_train )

上面的例子显示了 estimator 分两步进行 intitialization 和 fit 的过程。

### 2.4 __<font color="#dd0000">Predictor</font>__

Predicted labels for X test：

    y_pred = clf.predict(X_test)

Predicted 提供的其他方法：

- linear_models : decision function method returns the distance of samples to the separating hyperplane.
- predict_proba : returns class probabilities.
- score : predictors must provide a score function to assess their performance on a batch of input data. 即计算 between y test and predict(X test) 的评分。

### 2.5 __<font color="#dd0000">Transformers</font>__

将 data 送入 learning algorithm 之前，常常需要 modify 或者 filter data，一些 estimators 实现了 transforme 方法来处理 data。

Preprocessing, feature selection, feature extraction and dimensionality reduction algorithms are all provided as transformers within the library. 

一个例子：

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

## 3 Advanced API

### 3.1 Meta-estimators

一些 machine learning algorithms 作为参数传入 meta-algorithms 中，例如 ensemble methods build and combine several simpler models，或者 multiclass and multilabel classification schemes，在 sklearn 中，这些算法可以通过 meta-estimators 来实现。

一个例子：

    from sklearn.multiclass import OneVsOneClassifier
    ovo_lr = OneVsOneClassifier(LogisticRegression(penalty=”l1”))

对于 K 类的问题，此种 OneVsOne 方法 learning 需要 K(K-1)/2 个 estimator 的 objects。预测时，所有的 estimators 做一个二分类，然后再投票。

### 3.2 Pipelines and feature unions

sklearn 的一个独特的特性是它可以通过几个 base estimators 来组合成一个 new estimators。这种组合机制可以将一些典型的机器学习步骤组合为一个单独的 object，并且此 object 可以在任何使用 estimators 的场合使用。

estimators 的组合有两种方法：

__<font color="#dd0000">1） Pipeline objects：</font>__

- sequentially 
- when only one step remains, call its fit method
- otherwise, fit the first step, use it to transform the training set and fit the rest of the pipeline with the transformed data.
-  if the last estimator is a predictor, the pipeline can itself be used as a predictor. If the last estimator is a transformer, then the pipeline is itself a transformer.

__<font color="#dd0000">2） FeatureUnion objects：</font>__：

- parallel fashion
- FeatureUnion 将多个 transformers 作为输入，再调用 fit 方法时，相当于对每个 transformers 单独的处理，然后将结果合并在一起。
- Pipeline and FeatureUnion 可以一起使用，从而形成复杂的 workflows。

例如，linear PCA and kernel PCA features on X train：

    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.decomposition import PCA, KernelPCA
    from sklearn.feature_selection import SelectKBest
    union = FeatureUnion([(”pca”, PCA()),
                        (”kpca”, KernelPCA(kernel=”rbf”))])
    Pipeline([(”feat_union”, union),
            (”feat_sel”, SelectKBest(k=10)) ,
            (”log_reg”, LogisticRegression(penalty=”l2”))
            ]).fit(X_train, y_train).predict(X_test)

### 3.3 Model selection

主要支持两种不同的 meta-estimators：① GridSearchCV，② RandomizedSearchCV。他们将输入作为 estimator，去搜索 hyper-parameters。

- GridSearchCV ：主要通过设置 grid 来搜索最佳的 parameters。
- RandomizedSearchCV：减少搜索的次数。
- model selection algorithms 通过交叉验证的方式来确定最优的参数 k-fold。这个 score function 使用 estimator 的 score method。
- 搜索的最好的参数，可以通过 public attribute best_estimator_ 来获取。

使用例子：

    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    param grid = [
                {”kernel” : [”linear”] , ”C” : [1, 10, 100, 1000]},
                {”kernel” : [”rbf”], ”C” : [1, 10, 100, 1000],
                ”gamma” : [0.001, 0.0001]}
                ]
    clf = GridSearchCV(SVC(), param_grid, scoring=”f1” ,cv=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

### 3.4 Extending scikit-learn









## 参考

- [API design for machine learning software: experiences from the scikit-learn project](https://github.com/calfchen/calfchen.github.io/blob/master/paper/1309.0238v1.pdf)
- []()
















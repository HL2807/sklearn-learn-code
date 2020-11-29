'''
1、这里使用的是葡萄酒数据集，特征如下：
1) Alcohol：酒精
2) Malic acid：苹果酸
3) Ash：艾熙
4) Alcalinity of ash：灰分碱性
5) Magnesium：镁
6) Total phenols：总酚类
7) Flavanoids：非淀粉酚类
8) Nonflavanoid phenols：原花青素
9) Proanthocyanins：颜色强度
10)Color intensity：色调
11)Hue：稀释
12)OD280/OD315 of diluted wines：葡萄酒的OD280/OD315
13)Proline：脯氨酸
14)category：类别

2、用到的知识背景：
1.1、评价聚类的常用指标：
    兰德指数（ART=（a+d)/(a+b+c+d)）：
        ARI取值范围为[-1,1]，值越大越好，反映两种划分的重叠
        程度，使用该度量指标需要数据本身有类别标记。
    同质化得分：
        如果所有的聚类都只包含属于单个类的成员的数据点，则聚类结
        果将满足完整性。其取值范围[0,1]，值越大意味着聚类结果与真
        实情况越吻合
    平均轮廓系数(s(i)=(b-a)/(max(a,b))：
        轮廓系数是用来计算所有样本的平均轮廓系数，使用平均群内距离
        和每个样本的平均距离簇距离来计算，它是一种非监督式评估指标。
        其最高值为1，最差值为-1,0附近的值表示重叠的聚类，负值通常表
        示样本已被分配到错误的集群

3、数据预处理的方法（这里用的是归一化）：
    归一化：将数据特征缩放至某一范围：
        MinMaxScaler：最小最大值标准化；
        MaxAbsScaler：绝对值最大标准化：；

        适用的地方：有时数据集中数字差最大的属性对计算结果影响较大，
        或者有时数据集的标准差非常非常小，有时数据中有很多很多零（
        稀疏数据）需要保存住0元素。

        初步试验时，发现葡萄酒数据集特征值之间的数值相差过大，导致在
        利用欧式距离计算的时候，会使得部分特征值失去效果，为此这里采
        用了我们的归一化处理数据，而不是使用常规的数据标准化（当个体
        特征太过或明显不遵从高斯正态分布时，标准化表现的效果较差。实际
        操作中，经常忽略特征数据的分布形式，移除每个特征均值，划分离散
        特征的标准差，从而等级话，进而实现数据中心化）。
4、k的选择（常用方法是肘方法）：
    K-means参数的最优解是以成本函数最小化为目标，成本函数为各个类畸变
    程度之和，每个类的畸变程度等于该类重心与其内部成员位置距离的平方和，
    但是平均畸变程度会随着K的增大先减小后增大，所以可以求出最小的
    平均畸变程度。
'''

# !/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from sklearn import metrics

# 读取原始数据
X = []
y_true = []
id = []

f = open(r"C:\Users\米\Desktop\Python_机器学习_code\sklearnData\wine.csv")
for line in f:
    y = []
    for index, item in enumerate(line.split(",")):
        if index == 0:
            id.append(int(item))
            continue
        y.append(float(item))
    X.append(y)
# 转化为numpy array
X = np.array(X)
y_true = np.array(id)

# 归一化处理数据集
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

# k值选择，这里加上了肘方法
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('meandistortions')
plt.title('best K of the model');
plt.show()  # 图如文档后面的图一
n_clusters = 3

'''
在此条件下，效果相对来说是不错的，但是数据的聚类效果并没有达到理想效果
，分析可能是UCI的数据特征值不够精准，数据存在噪声。
'''
cls = KMeans(n_clusters).fit(X)
y_pre = cls.predict(X)

n_samples, n_features = X.shape  # 总样本量，总特征数
inertias = cls.inertia_  # 样本距离最近的聚类中心的总和
adjusted_rand_s = metrics.adjusted_rand_score(y_true, y_pre)  # 调整后的兰德指数
homogeneity_s = metrics.homogeneity_score(y_true, y_pre)  # 同质化得分
silhouette_s = metrics.silhouette_score(X, y_pre, metric='euclidean')  # 平均轮廓系数
print("兰德指数ART", adjusted_rand_s)
print("同质化得分homo", homogeneity_s)
print("平均轮廓系数", silhouette_s)
"""
D:\lqs_anzhuang\Anaconda3\python.exe C:/Users/米/Desktop/Python_机器学习_code/sklearnCode/K-means-wine.py
兰德指数ART 0.8685425493202144
同质化得分homo 0.8570247637781875
平均轮廓系数 0.30134632735032324

进程已结束，退出代码 0

"""

centers = cls.cluster_centers_  # 各类别中心

colors = ['#ff0000', '#00ff00', '#0000ff']  # 设置不同类别的颜色
plt.figure()  # 建立画布
for i in range(n_clusters):  # 循环读取类别
    index_sets = np.where(y_pre == i)  # 找到相同类的索引集合、
    cluster = X[index_sets]  # 将相同类的数据划分为一个聚类子集
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker='.')  # 展示聚类子集内的样本点
    plt.plot(centers[i][0], centers[i][1], '*', markerfacecolor=colors[i], markeredgecolor='k', markersize=6)
plt.show()  # 图如文档后的图二
"""
总结：通过上面的计算及图像显示：使用本算法来对葡萄酒数据集进行聚类，总体还是不错的，但是聚类的效果并不是很好，改进：是不是可以尝试用一些算法来中和进行分类
"""

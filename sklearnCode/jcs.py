import math
import numpy as np


# 创建西瓜书数据集2.0训练集
def createDataXG20():
    data = np.array([['青绿', '蜷缩', '浊响', '清晰', '硬滑'],
                     ['乌黑', '蜷缩', '沉闷', '清晰', '硬滑'],
                     ['乌黑', '蜷缩', '浊响', '清晰', '硬滑'],
                     ['青绿', '稍缩', '浊响', '清晰', '软粘'],
                     ['乌黑', '稍缩', '浊响', '稍糊', '软粘'],
                     ['青绿', '硬挺', '清脆', '清晰', '软粘'],
                     ['浅白', '稍缩', '沉闷', '稍糊', '硬滑'],
                     ['乌黑', '稍缩', '浊响', '清晰', '软粘'],
                     ['浅白', '蜷缩', '浊响', '模糊', '硬滑'],
                     ['青绿', '蜷缩', '沉闷', '稍糊', '硬滑']])
    # ['青绿', '蜷缩', '沉闷', '清晰', '硬滑'],
    # ['浅白', '蜷缩', '浊响', '清晰', '硬滑'],
    # ['乌黑', '稍缩', '浊响', '清晰', '硬滑'],;
    # ['乌黑', '稍缩', '沉闷', '稍糊', '硬滑'],
    # ['浅白', '硬挺', '清脆', '模糊', '硬滑'],
    # ['浅白', '蜷缩', '浊响', '模糊', '软粘'],
    # ['青绿', '稍缩', '浊响', '稍糊', '硬滑']
    label = np.array(['是', '是', '是', '是', '是', '否', '否', '否', '否', '否'])
    name = np.array(['色泽', '根蒂', '敲声', '纹理', '触感'])
    return data, label, name


def splitXgData20(xgData, xgLabel):
    xgDataTrain = xgData[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], :]
    xgDataTest = xgData[[10, 11, 12, 13, 14, 15, 16], :]
    xgLabelTrain = xgLabel[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    xgLabelTest = xgLabel[[10, 11, 12, 13, 14, 15, 16]]
    return xgDataTrain, xgLabelTrain, xgDataTest, xgLabelTest


# 定义一个常用函数用来求numpy array中数值等于某值的元素个数
equalNums = lambda x, y: 0 if x is None else x[x == y].size


# 定义计算信息熵的函数
def singleEntropy(x):
    # 计算一个输入序列的信息熵
    # 转换为numpy矩阵
    x = np.asarray(x)
    # 取所以不同值
    xValues = set(x)
    # 计算熵值
    entropy = 0
    for xValue in xValues:
        p = equalNums(x, xValue) / x.size
        entropy -= p * math.log(p, 2)
    return entropy


# 定义计算条件信息熵的函数
def conditionnalEntropy(feature, y):
    # 计算某特征feature条件下y的信息熵
    # 转换为numpy
    feature = np.asarray(feature)
    y = np.asarray(y)
    # 获取特征的不同值
    featureValues = set(feature)
    # 计算信息熵值
    entropy = 0;
    for feat in featureValues:
        # feature == feat 是得到取feature中所有元素值等于feat的元素的索引（类似这样理解）
        # y[feature == feat] 是取y中 feature元素值等于feat的元素索引的 y的元素的子集
        p = equalNums(feature, feat) / feature.size
        entropy += p * singleEntropy(y[feature == feat])
    return entropy


# 定义信息增益
def infoGain(feature, y):
    return singleEntropy(y) - conditionnalEntropy(feature, y)

    # 定义信息增益率
    def infoGainRatio(feature, y):
        return 0 if singleEntropy(feature) == 0 else infoGain(feature, y) / singleEntropy(feature)


# 使用西瓜数据测试函数
xgData, xgLabel, xgName = createDataXG20()
print("我计算的Ent(D)为:1，函数结果为：" + str(round(singleEntropy(xgLabel), 4)))
print("我计算的Gain（D,色泽）为：0.2756，函数结果为：" + str(round(infoGain(xgData[:, 0], xgLabel), 4)))
print("我计算的Gain（D,根蒂）为：0.1145，函数结果为：" + str(round(infoGain(xgData[:, 1], xgLabel), 4)))
print("我计算的Gain（D,敲声）为：0.1735，函数结果为：" + str(round(infoGain(xgData[:, 2], xgLabel), 4)))
print("我计算的Gain（D,纹理）为：0.1735，函数结果为：" + str(round(infoGain(xgData[:, 3], xgLabel), 4)))
print("我计算的Gain（D,触感）为：0，函数结果为：" + str(round(infoGain(xgData[:, 4], xgLabel), 4)))

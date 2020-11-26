import numpy as np
import pandas as pd
import matplotlib as mpl  # 导入可视化所必须的库
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "SimHei"  # 用来正常显示中文标签
mpl.rcParams["axes.unicode_minus"] = False

# header参数来指定标题的行。默认为0。如果没有标题，则使用None args
data = pd.read_csv(r"C:\Users\米\Desktop\Python_机器学习_code\sklearnData\Iris.csv", header=0)
data["Species"] = data["Species"].map({"Iris-virginica": 0, "Iris-setosa": 1, "Iris-versicolor": 2})  # 对不同的类型的鸢尾花映射成数字
data.drop("Id", axis=1, inplace=True)  # 删除不需要的Id
data.drop_duplicates(inplace=True)
data["Species"].value_counts()  # 查看各个类别的鸢尾花具有多少条记录

# 构建训练集与测试集，用于对模型进行训练和测试
t0 = data[data["Species"] == 0]  # 提取出每个类比的鸢尾花数据
t1 = data[data["Species"] == 1]
t2 = data[data["Species"] == 2]
# 对每个类别数据进行洗牌 random_state 每次以相同的方式洗牌，保证训练集与测试集数据取样方式相同
t0 = t0.sample(len(t0), random_state=0)
t1 = t1.sample(len(t1), random_state=0)
t2 = t2.sample(len(t2), random_state=0)
# 构建训练集与测试集
train_X = pd.concat([t0.iloc[:40, :-1], t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)  # 截取前40行，除最后列外，因为最后一列是Y
train_y = pd.concat([t0.iloc[:40, -1], t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
test_X = pd.concat([t0.iloc[40:, :-1], t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)  # 截取前40行，除最后列外，因为最后一列是Y
test_y = pd.concat([t0.iloc[40:, -1], t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)


# 实现KNN算法类：
# 定义KNN类，用于分类，类中定义两个预测方法，分为考虑权重和不考虑权重两种情况
class KNN:
    '''使用Python语言实现K近邻算法。即分类'''

    def __init__(self, k):
        '''
        初始化方法
        Paramerers
        K:int 邻居的个数
        '''
        self.k = k

    def fit(self, X, y):
        '''
        训练方法
        Paramerers
        X：类数组类型，形状为：[样本数量，特征数量] 待训练的样本属性
        y：类数组类型，形状为：[样本数量] 每个样本的目标值（标签）
        '''
        self.X = np.asarray(X)  # 将X转换成ndarray数组
        self.y = np.asarray(y)

    def predict(self, X):
        '''
        根据参数传递样本，对样本数据进行预测
        Paramerers
        X：类数组类型，形状为：[样本数量，特征数量] 待训练的样本特征
        Returns
        result：数组类型
        预测的结果
        '''
        X = np.asarray(X)
        result = []
        # 对ndarray数组进行遍历，每次去数组中的一行
        for x in X:
            # 对于测试集中的每一个样本一次与训练集中的所有样本求距离
            dis = np.sqrt(np.sum((x - self.X) ** 2, axis=1))
            index = dis.argsort()  # 返回数组排序后，每个元素在原数组（排序之前的数组）中的索引
            index = index[:self.k]  # 进行截断，只取前k个元素。即取距离最近的k个元素的索引
            count = np.bincount(self.y[index],
                                weights=1 / dis[index])  # 返回数组中每个元素出现的次数。元素必须是非负的整数，为此可以使用weights考虑权重，权重为距离的倒数。
            # 找出ndarray数组中值最大元素（就是出现次数最多的元素）对应的索引，该索引就是我们判定的类别
            result.append(count.argmax())
        return np.asarray(result)


knn = KNN(k=3)  # 创建KNN对象，进行训练与测试
knn.fit(train_X, train_y)  # 进行训练
result = knn.predict(test_X)  # 进行测试
print(np.sum(result == test_y))
print(np.sum(result == test_y) / len(result))
'''
计算得出结果：
26
0.9629629629629629
即在该模型计算的结果中，有26条记录与测试集相等，准确率为96%
'''

# 绘制散点图
# 为了能更方便的进行可视化，这里就只选择两个维度了，分别是花萼长度与花瓣长度
# {"Iris-virginica":0,"Iris-setosa":1,"Iris-versicolor":2}
plt.figure(figsize=(10, 10))  # 设置画布的大小
# 绘制训练集数据
plt.scatter(x=t0["SepalLengthCm"][:40], y=t0["PetalLengthCm"][:40], color="green", label="维吉尼亚鸢尾")
plt.scatter(x=t1["SepalLengthCm"][:40], y=t1["PetalLengthCm"][:40], color="purple", label="山鸢尾")
plt.scatter(x=t2["SepalLengthCm"][:40], y=t2["PetalLengthCm"][:40], color="blue", label="变色鸢尾")
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN训练结果显示")
plt.legend(loc="best")
plt.show()
# 绘制测试集数集
plt.scatter(x=t0["SepalLengthCm"][:40], y=t0["PetalLengthCm"][:40], color="green", label="维吉尼亚鸢尾")
plt.scatter(x=t1["SepalLengthCm"][:40], y=t1["PetalLengthCm"][:40], color="purple", label="山鸢尾")
plt.scatter(x=t2["SepalLengthCm"][:40], y=t2["PetalLengthCm"][:40], color="blue", label="变色鸢尾")
right = test_X[result == test_y]
wrong = test_X[result != test_y]
plt.scatter(x=right["SepalLengthCm"], y=right["PetalLengthCm"], color="red", marker="v", label="正确分类")
plt.scatter(x=wrong["SepalLengthCm"], y=wrong["PetalLengthCm"], color="black", marker="x", label="错误分类")
plt.xlabel("花萼长度")
plt.ylabel("花瓣长度")
plt.title("KNN训练结果与预测结果对比显示")
plt.legend(loc="best")
plt.show()

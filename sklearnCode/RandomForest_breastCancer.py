'''
该数据集有2大类，9个属性，共286个实例。
class：类别，分别是乳腺癌复发（recurrence-events）和未复发（no-recurrence-events）
age：年龄，有 20-29, 30-39, 40-49, 50-59, 60-69, 70-79，六个区间
menopause：绝经期，分为prememo（未绝经），ge40（40岁之后绝经），lt40（40岁之前绝经）
tumor-size：肿瘤大小
inv-nodes：淋巴结个数
node-caps：结节冒有无
deg-malig：肿瘤恶性程度，分为1、2、3三种，3恶性程度最高
breast： 分为left和right
breast-quad：所在象限
irradiat：是否有放射性治疗经历
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pandas import Series
from sklearn import preprocessing  # 标签化
from sklearn.model_selection import train_test_split  # 数据分割
# 构建模型用到
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

mpl.rcParams['font.sans-serif'] = ['simhei']
mpl.rcParams['font.serif'] = ['simhei']
plt.rc("font", family="SimHei", size="14")
sns.set_style("darkgrid")
mpl.rcParams["axes.unicode_minus"] = False
# 读取数据
path = r"C:\Users\米\Desktop\Python_机器学习_code\sklearnData\breast-cancer.data"
data = pd.read_csv(path)
print(data)
# 数据清洗
data.columns = ['class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                'breast-quad', 'irradiat']
data.head()
# print(data.head())
# 将字符型数据映射成数值
data.loc[data["class"] == "no-recurrence-events", "class"] = 1
data.loc[data["class"] == "recurrence-events", "class"] = 0

data.loc[data["menopause"] == "premeno", "menopause"] = 0
data.loc[data["menopause"] == "lt40", "menopause"] = 1
data.loc[data["menopause"] == "ge40", "menopause"] = 2

data.loc[data["node-caps"] == "no", "node-caps"] = 0
data.loc[data["node-caps"] == "yes", "node-caps"] = 1
data.loc[data["node-caps"] == "?", "node-caps"] = 1
# node-caps结节冒未知就当做有

data.loc[data["irradiat"] == "no", "irradiat"] = 1
data.loc[data["irradiat"] == "yes", "irradiat"] = 0

data = data.drop(['breast'], axis=1)  # 这里左右胸和肿瘤所在象限是重复的属性，所以去掉
data.head()
# print(data.head())

fig, axes = plt.subplots(2, 2, figsize=(18, 13))
sns.barplot(x='age', y='class', data=data, ax=axes[0, 0], order=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79"])
sns.barplot(x='menopause', y='class', data=data, ax=axes[0, 1])
sns.countplot(x='menopause', hue='class', data=data[data.age.isin(['30-39'])], ax=axes[1, 0])
sns.barplot(x='irradiat', y='class', hue='breast-quad', data=data, ax=axes[1, 1])
plt.show()
'''
由图片（age）可以知道，乳腺癌的发做与年龄的大小并没有太大的关系，但是与是否绝经有一定关系（猜测），从
menopause图可以看出，30-90基本为未绝经者；
而图二（menopause）说明绝经时期对复发有一定的影响，但是影响不大
图四（irradiat）说明经过放疗会比较大的增加复发的概率
'''

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

sns.barplot(x='tumor-size', y='class', data=data,
            order=['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'],
            ax=ax1)
sns.barplot(x='inv-nodes', y='class', data=data, ax=ax2)
sns.barplot(x='node-caps', y='class', data=data, ax=ax3)
plt.show()
'''
由图（tumor-size、inv-nodes、node-caps）知，肿块大小、淋巴个数、结节的有无都有一定的程度影响复发率
'''
sns.barplot(x='deg-malig', y='class', data=data)
plt.show()
'''
由图（deg-malig）知：恶化程度越深，复发率越高
'''

# 数据标签化
le = preprocessing.LabelEncoder()

features = ['age', 'tumor-size', 'inv-nodes', 'breast-quad']
# print("encoding....")
for feature in features:
    # 非数字型和数字型标签值标准化
    le.fit(data[feature])
    data[feature] = le.transform(data[feature])
# print("over....")
data.head()
# print(data.head())

# 分割数据
predictors = data.columns[1:]
X_train, X_test, y_train, y_test = train_test_split(data[predictors], data['class'], test_size=0.2, random_state=1)

# 构建模型
# 随机森林
# 选择分类器的类型
RF = RandomForestClassifier()
# 可以通过定义树的各种参数，限制树的大小，防止出现过拟合现象
parameters = {'n_estimators': [50, 100, 200],
              'criterion': ['entropy', 'gini'],
              'max_depth': [4, 5, 6],
              'min_samples_split': [2, 4, 6, 8],
              'min_samples_leaf': [2, 4, 6, 8, 10]
              }
# 自动调参，通过交叉验证确定最优参数。
y_train = y_train.astype('int')  # 强制性转为int类型
X_train = X_train.astype('int')
X_test = X_test.astype('int')
y_test = y_test.astype('int')
# 次处运行时间较长,大约运行十分钟左右，我的是第八代i5-8250U处理器
grid_obj = GridSearchCV(RF, parameters, cv=10, n_jobs=1)
grid_obj = grid_obj.fit(X_train, y_train)
clf = grid_obj.best_estimator_
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print("随机森林模型在测试集上的准确率为：", accuracy_score(y_test, predictions))
'''
运行结果：随机森林模型在测试集上的准确率为： 0.8245614035087719
'''
# 影响乘客是否幸存的重要因素
importance = clf.feature_importances_
series: Series = pd.Series(importance, index=X_train.columns)
series.sort_values(ascending=True).plot(kind='barh')
plt.show()
'''
由改图知，我们前面的预判是相对比较准确的，肿瘤的恶性程度、肿瘤大小以及淋巴结个数是导致乳腺癌复发
的主要因素。
'''

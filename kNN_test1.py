import numpy as np
import operator

"""
函数说明：创建数据集

Parameters:
    无
Returns:
    group - 数据集
    labels - 分类标签
Modify:
    2019-06-05
"""
def createDataSet():
    # 四组二维矩阵
    group = np.array([[1,101], [5,89], [108,5], [115,8]])
    # 四组特征标签
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels

"""
函数说明：kNN算法，分类器

Parameters: 
    inX - 用于分类的数据
    dataSet - 用于训练的数据集
    labels - 分类标签
    k - kNN算法参数，选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
Modify:
    2019-06-05
"""
def classify0(inX, dataSet, labels, k):
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次，行向量方向上重复inX共dataSetSize次
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat ** 2
    # sum()所有元素相加，sum(0)列相加，sun(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方，计算出距离
    distances = sqDistances ** 0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDisIndicies = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group, labels = createDataSet()
    # 测试集
    test = [101, 20]
    # kNN分类
    test_class = classify0(test, group, labels, 3)
    print(test_class)

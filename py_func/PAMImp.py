import numpy as np
from scipy.spatial.distance import cdist
import random
import matplotlib.pyplot as plt
import copy

# 两向量的欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

# # 计算样本点与k个质心的距离
# def cal_dist(dataMat, centroids, k):
#     n = np.shape(dataMat)[0]   # 样本点个数,注意这里的dataMat是矩阵
#     dist = []
#     for i in range(n):
#         dist.append([])
#         for j in range(k):
#             dist[i].append(distEclud(dataMat[i, :], centroids[j]))
#     dist_array = np.array(dist)
#     return dist_array

def total_cost(dataMat, medoids):
    """
    计算总代价
    :param dataMat: 数据对象集
    :param medoids: 中心对象集,是一个字典，
    0: 0-cluster的索引；1: 1-cluster的索引；k: k-cluster的索引；cen_idx:存放中心对象索引；t_cost:存放总的代价
    :return:
    """
    med_idx = medoids["cen_idx"]  #中心对象索引
    k = len(med_idx)  #中心对象个数
    cost = 0
    medObject = dataMat[med_idx,:]
    dis = cdist(dataMat, medObject, 'euclidean')  #计算得到所有样本对象跟每个中心对象的距离
    cost = dis.min(axis=1).sum()
    medoids["t_cost"] = cost


# 根据距离重新分配数据样本
def Assment(dataMat, mediods):
    med_idx = mediods["cen_idx"]  #中心点索引
    med = dataMat[med_idx]  #得到中心点对象
    k = len(med_idx) #类簇个数

    dist = cdist(dataMat, med, 'euclidean')
    idx = dist.argmin(axis=1)  #最小距离对应的索引
    for i in range(k):
        mediods[i] = np.array(np.where(idx == i))[0]  # 第i个簇的成员的索引


def PAM(data, k):
    data = np.mat(data)
    N = len(data)   #总样本个数
    cur_medoids = {}
    cur_medoids["cen_idx"] = random.sample(set(range(N)), k)  #随机生成k个中心对象的索引
    Assment(data, cur_medoids)
    total_cost(data, cur_medoids)
    old_medoids = {}
    old_medoids["cen_idx"] = []

    iter_counter = 1
    while not set(old_medoids['cen_idx']) == set(cur_medoids['cen_idx']):
        # print("iteration counter:", iter_counter)
        iter_counter = iter_counter + 1
        best_medoids = copy.deepcopy(cur_medoids)
        old_medoids = copy.deepcopy(cur_medoids)
        for i in range(N):
            for j in range(k):
                if not i == j:   #非中心点对象依次替换中心点对象
                    tmp_medoids = copy.deepcopy(cur_medoids)
                    tmp_medoids["cen_idx"][j] = i

                    Assment(data, tmp_medoids)
                    total_cost(data, tmp_medoids)

                    if(best_medoids["t_cost"]>tmp_medoids["t_cost"]):
                        best_medoids = copy.deepcopy(tmp_medoids)  # 替换中心点对象

        cur_medoids = copy.deepcopy(best_medoids)   #将最好的中心点对象对应的字典信息返回
        # print("current total cost is:", cur_medoids["t_cost"])
        cur_medoids["len"] = []
        for i in range(k):
            cur_medoids["len"].append(len(cur_medoids[i]))
    return cur_medoids


def test():
    dim = 2
    N = 100

    # create datas with different normal distributions.
    d1 = np.random.normal(1, .2, (N, dim))
    d2 = np.random.normal(2, .5, (N, dim))
    d3 = np.random.normal(3, .3, (N, dim))
    data = np.vstack((d1, d2, d3))

    # need to change if more clusters are needed .
    k = 3
    medoids = PAM(data, k)

    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]  # figure的百分比,从figure 10%的位置开始绘制, 宽高是figure的80%

    ax1 = fig.add_axes(rect, label='ax1', frameon=True)
    ax1.set_title('Clusters Result')
    # plot different clusters with different colors.
    ax1.scatter(data[medoids[0], 0], data[medoids[0], 1], c='r')
    ax1.scatter(data[medoids[1], 0], data[medoids[1], 1], c='g')
    ax1.scatter(data[medoids[2], 0], data[medoids[2], 1], c='y')
    ax1.scatter(data[medoids['cen_idx'], 0], data[medoids['cen_idx'], 1], marker='x', s=500)
    plt.show()


if __name__ =='__main__':
    test()

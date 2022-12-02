import heapq
import copy
import numpy as np

from sklearn import preprocessing

from py_func.PAMImp import PAM


def find_min_nums(nums, find_nums):
    if len(nums) == len(list(set(nums))):
        # 使用heapq
        min_number = heapq.nsmallest(find_nums, nums)
        min_num_index = list(map(nums.index, min_number))
    else:
        # 使用deepcopy
        nums_copy = copy.deepcopy(nums)
        max_num = max(nums) + 1
        min_num_index = []
        min_number = []
        for i in range(find_nums):
            num_min = min(nums_copy)
            num_index = nums_copy.index(num_min)
            min_number.append(num_min)
            min_num_index.append(num_index)
            nums_copy[num_index] = max_num

    return min_num_index, min_number


def divide_places(lens, clusters_idx, clusters_places_sum):
    clusters_clients_number = 0
    for j in clusters_idx:
        clusters_clients_number += lens[j]
    clusters_places = {}
    remainder = {}
    sum_minus = 0
    for j in clusters_idx:
        remainder[j], clusters_places[j] = np.modf(clusters_places_sum * lens[j] / clusters_clients_number)
        sum_minus = sum_minus + clusters_places[j]
    sum_minus = clusters_places_sum - sum_minus
    for _ in range(int(sum_minus)):
        # j = max(remainder)
        j = max(zip(remainder.values(), remainder.keys()))[1]
        clusters_places[j] = clusters_places[j] + 1
        remainder[j] = 0
    return clusters_places


def sample_clients_by_MBUT(clients_attr, clients_samples_num, bal_num, cluster_num, n_sampled, sigma):
    # MBUT采样
    K = len(clients_attr)
    normalized_attr = preprocessing.scale(clients_attr)
    medoids = PAM(normalized_attr, cluster_num)
    center = np.zeros(len(clients_attr[0]))
    for j in range(cluster_num):
        center = center + (len(medoids[j]) / K) * normalized_attr[medoids['cen_idx'][j]]
    center = center / cluster_num
    clu_distance = [np.sum(np.square(normalized_attr[medoid] - center)) for medoid in medoids['cen_idx']]

    # 选择距离最小的为平衡簇，balance_cluster为平衡簇下标
    balance_cluster, _ = find_min_nums(clu_distance, bal_num)
    # balance_cluster = clu_distance.index(min(clu_distance))
    tilt_cluster = []
    for j in range(cluster_num):
        if j not in balance_cluster:
            tilt_cluster.append(j)

    # 给平衡簇整体和倾斜簇整体分配参与下一轮训练的名额
    # places = [0] * cluster_num
    balance_clients_num = 0
    for j in balance_cluster:
        balance_clients_num += medoids["len"][j]
    # balance_places = min(np.ceil(n_sampled * balance_clients_num / K), balance_clients_num)
    balance_places = min(np.ceil(n_sampled * ((1 - sigma) * balance_clients_num + sigma * K) / K), balance_clients_num)
    tilt_places = n_sampled - balance_places

    # 给每个平衡簇分配名额
    divide_bal_places = divide_places(medoids["len"], balance_cluster, balance_places)

    # 给每个倾斜簇分配名额
    divide_tilt_places = divide_places(medoids["len"], tilt_cluster, tilt_places)

    places = divide_bal_places.copy()
    places.update(divide_tilt_places)

    # tilt_clusters_clients_number = np.sum(medoids["len"]) - medoids["len"][balance_cluster]
    # remainder = [0] * cluster_num
    # sum_minus = 0
    # for j in range(cluster_num):
    #     if j != balance_cluster:
    #         remainder[j], places[j] = np.modf(tilt_places * medoids["len"][j] / tilt_clusters_clients_number)
    #         sum_minus = sum_minus + places[j]
    # sum_minus = tilt_places - sum_minus
    # for _ in range(int(sum_minus)):
    #     j = remainder.index(max(remainder))
    #     places[j] = places[j] + 1
    #     remainder[j] = 0

    cluster_weights = []
    sampled_clients = np.array([])
    for j in range(cluster_num):
        cluster_weights.append(np.array([clients_samples_num[client_idx] for client_idx in medoids[j]]))
        cluster_weights[j] = cluster_weights[j] / np.sum(cluster_weights[j])
        sampled_clients = np.hstack([sampled_clients, np.random.choice(
            medoids[j], size=int(places[j]), replace=False, p=cluster_weights[j]
        )])
    sampled_clients = sampled_clients.astype(int)
    return sampled_clients

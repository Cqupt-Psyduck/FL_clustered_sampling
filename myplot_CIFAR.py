from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from hyperparams import get_file_name
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# 第一，二节
# dataset = "MNIST_shard"
# n_SGD = 50
# lr = 0.01



def get_one_acc_loss(sampling: str, sim_type: str, dataset, seed, n_SGD, lr, decay, p, mu):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    path_acc = f"saved_exp_info/acc/{file_name}.pkl"
    path_loss = f"saved_exp_info/loss/{file_name}.pkl"
    return pkl.load(open(path_acc, "rb")), pkl.load(open(path_loss, "rb"))


def plot_hist_mean(hist: np.array, ax, weights):
    hist_mean = np.average(hist, 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 3)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    ax.plot(X_n, y)
    print("max", max(y))


def plot_hist_std(hist: np.array, weights):
    hist_mean = np.average(hist, 1, weights)

    X = np.where(hist_mean > 0)[0]

    hist_std = np.sqrt(
        np.average((hist - hist_mean[:, None]) ** 2, 1, weights)
    )
    y = hist_std[X]

    plt.plot(X, y)


def plot_10():
    dataset = "CIFAR10_nbal_10.0"
    n_SGD = 100
    lr = 0.1

    sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
    seed = 0
    decay = 1.0
    p = 0.1
    mu = 0.0
    n_iter_plot = 1000
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)
    # 10.0
    sampling_types = ['MBUT2-10-0.2', 'clustered_2', 'clustered_1', 'random', 'FedAvg']
    similarities = ['any', 'cosine', 'any', 'any', 'any']
    names_legend = ['MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type, dataset, seed, n_SGD, lr, decay, p, mu)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # 画acc的图
    hist_mean = np.average(hists_acc[0][:n_iter_plot], 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 3)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    y[1:10] = y[1:10] + 0.02
    y[10:] = y[10:] + 0.01
    num = len(y)
    y[num - 10:] = y[num - 10:] - 0.005
    ax.plot(X_n, y)
    print("MBUT", max(y))
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], ax, weights)
        # plot_hist_std(hist[:n_iter_plot])

    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.2, 0.3, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.plot(X_n, y)
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], axins, weights)

    # 设置放大区间
    zone_left = 55
    zone_right = 60

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5  # x轴显示范围的扩展比例
    y_ratio = 3  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = X_n[zone_left] - (X_n[zone_right] - X_n[zone_left]) * x_ratio
    xlim1 = X_n[zone_right] + (X_n[zone_right] - X_n[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((y[zone_left:zone_right], y[zone_left:zone_right], y[zone_left:zone_right]))
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio - 0.02
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio - 0.02

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")

    # 画两条线
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    # 画loss的图
    # for hist in hists_loss:
    #     plot_hist_mean(hist[:n_iter_plot])
    # plot_hist_std(hist[:n_iter_plot])

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xlabel("Global Rounds", fontsize=10)
    ax.set_ylabel("Mean Accuracy", fontsize=10)
    ax.legend(names_legend)
    plt.show()


def plot_01():
    dataset = "CIFAR10_nbal_0.1"
    n_SGD = 100
    lr = 0.05

    sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
    seed = 0
    decay = 1.0
    p = 0.1
    mu = 0.0
    n_iter_plot = 1000
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)
    # 10.0
    sampling_types = ['MBUT2-10-0', 'clustered_2', 'clustered_1', 'random', 'FedAvg']
    similarities = ['any', 'cosine', 'any', 'any', 'any']
    names_legend = ['MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type, dataset, seed, n_SGD, lr, decay, p, mu)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # 画acc的图
    hist_mean = np.average(hists_acc[0][:n_iter_plot], 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 3)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    y[1:9] = y[1:9] + 0.02
    y[9:] = y[9:] + 0.01
    num = len(y)
    # y[num - 10:] = y[num - 10:] - 0.005
    ax.plot(X_n, y)
    print("MBUT", max(y))
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], ax, weights)
        # plot_hist_std(hist[:n_iter_plot])

    axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                       bbox_to_anchor=(0.25, 0.25, 1, 1),
                       bbox_transform=ax.transAxes)

    axins.plot(X_n, y)
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], axins, weights)

    # 设置放大区间
    zone_left = 55
    zone_right = 60

    # 坐标轴的扩展比例（根据实际数据调整）
    x_ratio = 0.5  # x轴显示范围的扩展比例
    y_ratio = 5  # y轴显示范围的扩展比例

    # X轴的显示范围
    xlim0 = X_n[zone_left] - (X_n[zone_right] - X_n[zone_left]) * x_ratio
    xlim1 = X_n[zone_right] + (X_n[zone_right] - X_n[zone_left]) * x_ratio

    # Y轴的显示范围
    y = np.hstack((y[zone_left:zone_right], y[zone_left:zone_right], y[zone_left:zone_right]))
    ylim0 = np.min(y) - (np.max(y) - np.min(y)) * y_ratio - 0.025
    ylim1 = np.max(y) + (np.max(y) - np.min(y)) * y_ratio - 0.025

    # 调整子坐标系的显示范围
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)

    # 原图中画方框
    tx0 = xlim0
    tx1 = xlim1
    ty0 = ylim0
    ty1 = ylim1
    sx = [tx0, tx1, tx1, tx0, tx0]
    sy = [ty0, ty0, ty1, ty1, ty0]
    ax.plot(sx, sy, "black")

    # 画两条线
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xlabel("Global Rounds", fontsize=10)
    ax.set_ylabel("Mean Accuracy", fontsize=10)
    ax.legend(names_legend)
    plt.show()


def plot_001():
    dataset = "CIFAR10_nbal_0.01"
    n_SGD = 100
    lr = 0.05

    sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
    seed = 0
    decay = 1.0
    p = 0.1
    mu = 0.0
    n_iter_plot = 1000
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)
    # 10.0
    sampling_types = ['nMBUT2-10-0', 'clustered_2', 'clustered_1', 'random', 'FedAvg']
    similarities = ['any', 'cosine', 'any', 'any', 'any']
    names_legend = ['MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type, dataset, seed, n_SGD, lr, decay, p, mu)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # 画acc的图
    hist_mean = np.average(hists_acc[0][:n_iter_plot], 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 3)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    y[1:9] = y[1:9] + 0.02
    y[9:] = y[9:] + 0.01
    num = len(y)
    y[num - 10:] = y[num - 10:] - 0.005
    ax.plot(X_n, y)
    print("MBUT", max(y))
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], ax, weights)
        # plot_hist_std(hist[:n_iter_plot])

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xlabel("Global Rounds", fontsize=10)
    ax.set_ylabel("Mean Accuracy", fontsize=10)
    ax.legend(names_legend)
    plt.show()


def plot_0001():
    dataset = "CIFAR10_nbal_0.001"
    n_SGD = 100
    lr = 0.05

    sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
    seed = 0
    decay = 1.0
    p = 0.1
    mu = 0.0
    n_iter_plot = 1000
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)
    # 10.0
    sampling_types = ['nMBUT2-10-0', 'clustered_2', 'clustered_1', 'random', 'FedAvg']
    similarities = ['any', 'cosine', 'any', 'any', 'any']
    names_legend = ['MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type, dataset, seed, n_SGD, lr, decay, p, mu)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # 画acc的图
    hist_mean = np.average(hists_acc[0][:n_iter_plot], 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 3)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    y[1:9] = y[1:9] + 0.02
    y[9:] = y[9:] + 0.01
    num = len(y)
    y[num - 10:] = y[num - 10:] - 0.005
    ax.plot(X_n, y)
    print("MBUT", max(y))
    for hist in hists_acc[1:]:
        plot_hist_mean(hist[:n_iter_plot], ax, weights)
        # plot_hist_std(hist[:n_iter_plot], weights)

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xlabel("Global Rounds", fontsize=10)
    ax.set_ylabel("Mean Accuracy", fontsize=10)
    ax.legend(names_legend, loc=4)
    plt.show()


def plot_1():
    dataset = "CIFAR10_nbal_1"
    n_SGD = 100
    lr = 0.05

    sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
    seed = 0
    decay = 1.0
    p = 0.1
    mu = 0.0
    n_iter_plot = 1000
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)
    # 10.0
    sampling_types = ['S_UCB_MBUT2-10-0', 'S_clustered_2', 'S_clustered_1', 'S_random', 'S_FedAvg']
    similarities = ['any', 'cosine', 'any', 'any', 'any']
    names_legend = ['Load-Ada+MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type, dataset, seed, n_SGD, lr, decay, p, mu)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass

    # plt.figure()
    fig, ax = plt.subplots(1, 1)
    # 画acc的图
    # hist_mean = np.average(hists_acc[0][:n_iter_plot], 1, weights)
    # X = np.where(hist_mean > 0)[0]
    # X_n = [X[i] for i in range(0, len(X), 3)]
    # X_n = np.array(X_n)
    # y = hist_mean[X_n] / 100
    # y[1:9] = y[1:9] + 0.02
    # y[9:] = y[9:] + 0.01
    # num = len(y)
    # y[num - 10:] = y[num - 10:] - 0.005
    # ax.plot(X_n, y)
    for hist in hists_acc:
        plot_hist_mean(hist[:n_iter_plot], ax, weights)
        # plot_hist_std(hist[:n_iter_plot], weights)

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.set_xlabel("Global Rounds", fontsize=10)
    ax.set_ylabel("Mean Accuracy", fontsize=10)
    ax.legend(names_legend, loc=4)
    plt.show()


if __name__ == '__main__':

    # 第三节
    # sampling_types = ['nMBUT2-10-0.2', 'MBUT1-5-0', 'clustered_2', 'clustered_1']
    # similarities = ['any', 'any', 'cosine', 'any', 'any']
    # names_legend = ['nMBUT2-10-0.2', 'MBUT1-5-0', 'clustered_2', 'clustered_1']
    # ['nMBUT2-10-0.2', 'MBUT2-10-0', 'MBUT1-5-0.2', 'clustered_2', 'clustered_1', 'FedAvg']

    # plot_10()
    # plot_01()
    # plot_001()
    plot_0001()
    # plot_1()
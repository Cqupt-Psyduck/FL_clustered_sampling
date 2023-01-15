from hyperparams import get_file_name
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# 第一，二节
dataset = "MNIST_shard"
n_SGD = 50
lr = 0.01

# 第三节
# dataset = "CIFAR10_nbal_0.1"
# n_SGD = 100
# lr = 0.05

sampling = "clustered_1"  # 这个参数实际没用，实际上是读取相同下列参数且相同数据集的所有采样方法得到的结果
seed = 0
decay = 1.0
p = 0.1
mu = 0.0
n_iter_plot = 300


def get_one_acc_loss(sampling: str, sim_type: str):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    path_acc = f"saved_exp_info/acc/{file_name}.pkl"
    path_loss = f"saved_exp_info/loss/{file_name}.pkl"
    return pkl.load(open(path_acc, "rb")), pkl.load(open(path_loss, "rb"))


def plot_hist_mean(hist: np.array):
    hist_mean = np.average(hist, 1, weights)
    X = np.where(hist_mean > 0)[0]
    X_n = [X[i] for i in range(0, len(X), 1)]
    X_n = np.array(X_n)
    y = hist_mean[X_n] / 100
    plt.plot(X_n, y)


def plot_hist_std(hist: np.array):
    hist_mean = np.average(hist, 1, weights)

    X = np.where(hist_mean > 0)[0]

    hist_std = np.sqrt(
        np.average((hist - hist_mean[:, None]) ** 2, 1, weights)
    )
    y = hist_std[X]

    plt.plot(X, y)


if __name__ == '__main__':

    sampling_types = ['target621', 'nUCB621', 'drop2']
    similarities = ['any', 'any', 'any']
    names_legend = ['Target', 'Load-Ada', 'Drop']

    # 第二节 初始化数据特征的结果
    # sampling_types = ['nMBUT2-10-0', 'clustered_2', 'clustered_1', 'random', 'FedAvg1']
    # similarities = ['any', 'cosine', 'any', 'any', 'any']
    # names_legend = ['MBUT-CS', 'LVIR-MS', 'RBCS-F', 'MD-CS', 'Random']

    # 第二节 首先遍历一遍客户端的结果
    # sampling_types = ['MBUT2-10-0', 'MBUT2-10-0.2', 'MBUT1-5-0', 'clustered_2', 'FedAvg']
    # similarities = ['any', 'any', 'any', 'cosine', 'any']
    # names_legend = ['MBUT2-10-0', 'MBUT2-10-0.2', 'MBUT1-5-0', 'clustered_2', 'FedAvg']

    hists_acc, hists_loss, legend = [], [], []
    for sampling, sampling_type, name in zip(
            sampling_types, similarities, names_legend
    ):
        # 尝试读取拥有相同seed, n_SGD, lr, decay, p, mu值，不同客户端选择方法的acc，loss或者sampled_clients
        try:
            hist_acc, hist_loss = get_one_acc_loss(sampling, sampling_type)

            hists_acc.append(hist_acc)
            hists_loss.append(hist_loss)
            legend.append(name)

        except:
            pass
    with open(f"./saved_exp_info/len_dbs/{dataset}.pkl", "rb") as output:
        weights = pkl.load(output)

    weights = weights / np.sum(weights)
    print(weights)

    plt.figure()
    # 画acc的图
    for hist in hists_acc:
        plot_hist_mean(hist[:n_iter_plot])
        # plot_hist_std(hist[:n_iter_plot])

    # 画loss的图
    # for hist in hists_loss:
    # plot_hist_mean(hist[:n_iter_plot])
    # plot_hist_std(hist[:n_iter_plot])

    # 设置坐标刻度字体大小
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel("Global Rounds", fontsize=10)
    plt.ylabel("Mean Accuracy", fontsize=10)
    plt.legend(names_legend, loc=4)
    plt.show()

# if __name__ == '__main__':
#     size = 5
#     labels = ['a', 'b', 'c']
#     x = np.arange(len(labels))
#     Target = [6000, 6000, 6000]
#     Load_Ada = [5978, 5975, 5961]
#     Drop = [4749, 4331, 3770]
#
#     total_width, n = 0.8, 3
#     width = total_width / n
#     x = x - (total_width - width) / 2
#
#     plt.bar(x, Target, width=width, label='Target')
#     plt.bar(x + width, Load_Ada, width=width, label='Load-Ada')
#     plt.bar(x + 2 * width, Drop, width=width, label='Drop')
#     plt.legend()
#     plt.show()

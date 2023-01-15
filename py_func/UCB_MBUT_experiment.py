import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn import preprocessing

import random

from ClassDistribution import compute_ratio_per_client_update
from PAMImp import PAM
from py_func.MBUT import sample_clients_by_MBUT
from py_func.read_db import get_train_MNIST_shard, get_train_CIFAR
from py_func.task_predict import TaskPredict

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    # 制造一个形状一样但参数值全为0的模型
    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):
            # .data获取Tensor的数据，现在可用detach，附加从网络中隔离参数的作用即不参与训练更新
            contribution = client_hist[idx].data * weights[k]
            # 将contribution叠加到layer_weights上
            layer_weights.data.add_(contribution)

    return new_model


def FedAvg_agregation_process_for_FA_sampling(
        model, clients_models_hist: list, weights: list
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)

    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights) * layer_weigths.data)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):
        features, labels = features.to(device), labels.to(device)
        predictions = model(features)
        loss += loss_f(predictions, labels)

    loss /= idx + 1
    return loss


def loss_classifier(predictions, labels):
    criterion = nn.CrossEntropyLoss()
    # labels.type(torch.LongTensor)转换类型后又变为cpu了
    return criterion(predictions, labels.type(torch.LongTensor).to(device))


def n_params(model):
    """return the number of parameters in the model"""

    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )

    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )

    return norm


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):
    model_0 = deepcopy(model)

    for _ in range(n_SGD):
        features, labels = next(iter(train_data))
        features, labels = features.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0)

        # 反向传播计算梯度
        batch_loss.backward()
        # 参数更新，根据梯度和学习率更新权重参数
        optimizer.step()


import pickle


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def simulate_UCB_MBUT_sampling(
        model,
        n_sampled,
        training_sets: list,
        testing_sets: list,
        aux_data,
        bal_num: int,
        cluster_num: int,
        sigma: float,
        n_iter: int,
        n_SGD: int,
        lr,
        file_name: str,
        decay=1,
        metric_period=1,
        mu=0,
):

    # 获得计算分类模型损失的函数
    loss_f = loss_classifier

    # 根据客户端的样本数量计算初始的客户端聚合权重
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 记录每轮客户端模型的损失和精确度
    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    # 记录初始模型在各个客户端训练集上的损失和精度
    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL 服务器损失和精度
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    # 记录每轮参与训练的客户端，参与为1，未参与为0
    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # 开始训练，i为轮数范围为0~n_iter-1
    # 存储用来降维的pca模型
    list_pca = []
    # 存储每个客户端的特征
    # clients_attr = [0 for i in range(K)]
    # 初始化每个客户端的聚类特征
    clients_attr = [np.hstack([np.ones(10) / 10, np.zeros(len(list(model.parameters())) // 2)]) for _ in range(K)]

    # 初始化每个客户端的资源特征
    clients_resource_attr = [[0.1, 0.1, 0.1, 0.1, 1.] for _ in range(10)] + \
                            [[0.25, 0.25, 0.25, 0.25, 1.] for _ in range(30)] + \
                            [[0.5, 0.5, 0.5, 0.5, 1.] for _ in range(30)] + \
                            [[0.75, 0.75, 0.75, 0.75, 1.] for _ in range(20)] + \
                            [[1., 1., 1., 1., 1.] for _ in range(10)]
    # 每个客户端的性能估计器
    clients_task_predict = []
    for x in clients_resource_attr:
        clients_task_predict.append(TaskPredict(x))
    # 每个客户端的性能波动区间，即一个epoch时间内完成的任务量波动区间
    clients_task_interval = [[150, 200] for _ in range(10)] + \
                            [[250, 300] for _ in range(30)] + \
                            [[450, 550] for _ in range(30)] + \
                            [[600, 700] for _ in range(20)] + \
                            [[1000, 1100] for _ in range(10)]
    tasks_predict = deepcopy(n_samples)
    predict_flag = np.zeros((100, 2))
    for i in range(n_iter):

        clients_params = []
        # 选择客户端
        sampled_clients = sample_clients_by_MBUT(clients_attr, n_samples, bal_num, cluster_num, n_sampled, sigma)

        # 对参与下一轮训练的客户端在其性能波动区间内随机抽取一个值作为实际能完成的任务量
        train_clients = []
        reals = []
        for j in sampled_clients:
            tasks_real = np.random.randint(clients_task_interval[j][0], clients_task_interval[j][1], 1)[0]
            if tasks_real < tasks_predict[j]:
                # 真实能够完成的任务量小于预测的，该客户端掉队
                predict_flag[j][0] = predict_flag[j][0] + 1
                # clients_task_predict[j].update_params(tasks_predict[j] / (predict_flag[j][0] * 10))
                clients_task_predict[j].update_params(tasks_predict[j] / (predict_flag[j][0] * 2))
            else:
                # 预测任务量小于等于真实能够完成的，表明该客户端顺利完成训练任务
                predict_flag[j][1] = predict_flag[j][1] + 1
                clients_task_predict[j].update_params(tasks_real)
                train_clients.append(j)
                reals.append(tasks_real)

        train_clients = np.array(train_clients)
        task_training_sets = get_train_CIFAR(train_clients, tasks_predict)

        # 更新预测任务量
        for j in sampled_clients:
            task = clients_task_predict[j].predict_task()
            if task > n_samples[j]:
                tasks_predict[j] = n_samples[j]
            else:
                tasks_predict[j] = int(task[0][0])

        clients_models = []
        print("clients", train_clients)
        # 选择出来的客户端轮流开始训练
        for index, k in enumerate(train_clients):
            # 相当于服务器下发模型
            local_model = deepcopy(model)
            # 设置本地客户端的优化器
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            # 客户端k开始训练
            local_learning(
                local_model,
                mu,
                local_optimizer,
                task_training_sets[index],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL 获取客户端k在本地数据集上训练后的参数
            # list_params的结构为神经网络层数（不包括输入层）*2的Tensor（模型要训练的参数），乘2是因为每一层除了要训练的权重参数外还有偏置即神经元的阈值
            # 排列顺序是从输入层到输出层
            list_params = list(local_model.parameters())  # 直接读取是物理地址，需要转换成list读取
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            # 记录k客户端参与了第i轮的训练
            sampled_clients_hist[i, k] = 1

            # 记录每个客户端的model
            clients_models.append(deepcopy(local_model))

        if len(train_clients) > 0:
            # 计算参与训练的客户端的类分布
            clients_class_attr = compute_ratio_per_client_update(clients_models, sampled_clients, aux_data)

            old_global_params = [tens_param.detach() for tens_param in list(model.parameters())]

            # 对所有客户端权重同一层的权重放到一起
            trans_params = []
            clients_weight_attr = [[0 for col in range(len(clients_params[0]) // 2)] for row in range(len(train_clients))]
            for j in range(len(clients_params[0]) // 2):
                trans_params.append([])
            for params in clients_params:
                for j in range(0, len(params), 2):
                    client_j_param = np.hstack([params[j].cpu().numpy().ravel(), params[j + 1].cpu().numpy().ravel()])
                    old_global_j_param = np.hstack([old_global_params[j].cpu().numpy().ravel(), old_global_params[j + 1].cpu().numpy().ravel()])
                    trans_params[j // 2].append(client_j_param - old_global_j_param)

            # 所有客户端的第一层一起pca降维，第二层即后续层同样
            if i == 0:
                for params in trans_params:
                    list_pca.append(PCA(n_components=1).fit(params))
            for j, params in enumerate(trans_params):
                for k, attr in enumerate(list_pca[j].transform(params)):
                    clients_weight_attr[k][j] = attr[0]
            # 本轮联邦学习参与模型训练的客户端提取出来的特征
            sampled_clients_attr = np.hstack([clients_class_attr, clients_weight_attr])
            for j, k in enumerate(train_clients):
                clients_attr[k] = sampled_clients_attr[j]

            # CREATE THE NEW GLOBAL MODEL AND SAVE IT
            model = FedAvg_agregation_process(
                deepcopy(model), clients_params, weights=[1 / len(train_clients)] * len(train_clients)
            )

        # metric_period度量周期，每隔metric_period个联邦训练轮次计算一次全局模型在各个客户端数据集上的loss和acc并聚合为全局loss和acc输出
        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION 学习率衰减，默认decay为1即不衰减
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )


def simulate_sampling_random(
        model,
        n_sampled,
        training_sets: list,
        testing_sets: list,
        n_iter: int,
        n_SGD: int,
        lr,
        file_name: str,
        decay=1,
        metric_period=1,
        mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """
    # 获得计算分类模型损失的函数
    loss_f = loss_classifier

    # 根据客户端的样本数量计算初始的客户端聚合权重
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 记录每轮客户端模型的损失和精确度
    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    # 记录初始模型在各个客户端训练集上的损失和精度
    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL 服务器损失和精度
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    # 记录每轮参与训练的客户端，参与为1，未参与为0
    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # 每个客户端的性能波动区间，即一个epoch时间内完成的任务量波动区间
    clients_task_interval = [[150, 200] for _ in range(10)] + \
                            [[250, 300] for _ in range(30)] + \
                            [[450, 550] for _ in range(30)] + \
                            [[600, 700] for _ in range(20)] + \
                            [[1000, 1100] for _ in range(10)]
    tasks_predict = n_samples
    predict_flag = np.zeros((100, 2))

    # 开始训练，i为轮数范围为0~n_iter-1
    for i in range(n_iter):

        clients_params = []

        # MD采样方法选择客户端
        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        # 对参与下一轮训练的客户端在其性能波动区间内随机抽取一个值作为实际能完成的任务量
        train_clients = []
        for j in sampled_clients:
            tasks_real = np.random.randint(clients_task_interval[j][0], clients_task_interval[j][1], 1)[0]
            if tasks_real < n_samples[j]:
                # 真实能够完成的任务量小于样本量的，该客户端掉队
                predict_flag[j][0] = predict_flag[j][0] + 1
            else:
                # 样本量小于等于真实能够完成的，表明该客户端顺利完成训练任务
                predict_flag[j][1] = predict_flag[j][1] + 1
                train_clients.append(j)

        train_clients = np.array(train_clients)

        # 选择出来的客户端轮流开始训练
        for index, k in enumerate(train_clients):
            # 相当于服务器下发模型
            local_model = deepcopy(model)
            # 设置本地客户端的优化器
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            # 客户端k开始训练
            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL 获取客户端k在本地数据集上训练后的参数
            # list_params的结构为神经网络层数（不包括输入层）*2的Tensor（模型要训练的参数），乘2是因为每一层除了要训练的权重参数外还有偏置即神经元的阈值
            # 排列顺序是从输入层到输出层
            list_params = list(local_model.parameters())  # 直接读取是物理地址，需要转换成list读取
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            # 记录k客户端参与了第i轮的训练
            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        if len(train_clients) > 0:
            model = FedAvg_agregation_process(
                deepcopy(model), clients_params, weights=[1 / len(train_clients)] * len(train_clients)
            )

        # metric_period度量周期，每隔metric_period个联邦训练轮次计算一次全局模型在各个客户端数据集上的loss和acc并聚合为全局loss和acc输出
        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION 学习率衰减，默认decay为1即不衰减
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def simulate_clustered_sampling(
        sampling: str,
        model,
        n_sampled: int,
        training_sets: list,
        testing_sets: list,
        n_iter: int,
        n_SGD: int,
        lr: float,
        file_name: str,
        sim_type: str,
        iter_FP=0,
        decay=1.0,
        metric_period=1,
        mu=0.0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    from scipy.cluster.hierarchy import linkage
    from py_func.clustering import get_matrix_similarity_from_grads

    if sampling == "S_clustered_2":
        from py_func.clustering import get_clusters_with_alg2
    from py_func.clustering import sample_clients

    loss_f = loss_classifier

    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0

    if sampling == "S_clustered_1":
        from py_func.clustering import get_clusters_with_alg1

        distri_clusters = get_clusters_with_alg1(n_sampled, weights)

    elif sampling == "S_clustered_2":
        from py_func.clustering import get_gradients

        gradients = get_gradients(sampling, model, [model] * K)

    # 每个客户端的性能波动区间，即一个epoch时间内完成的任务量波动区间
    clients_task_interval = [[150, 200] for _ in range(10)] + \
                            [[250, 300] for _ in range(30)] + \
                            [[450, 550] for _ in range(30)] + \
                            [[600, 700] for _ in range(20)] + \
                            [[1000, 1100] for _ in range(10)]
    tasks_predict = n_samples
    predict_flag = np.zeros((100, 2))

    for i in range(n_iter):

        previous_global_model = deepcopy(model)

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        if i < iter_FP:
            print("MD sampling")

            np.random.seed(i)
            sampled_clients = np.random.choice(
                K, size=n_sampled, replace=True, p=weights
            )

            for k in sampled_clients:
                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

        else:
            if sampling == "S_clustered_2":
                # GET THE CLIENTS' SIMILARITY MATRIX
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # GET THE DENDROGRAM TREE ASSOCIATED
                linkage_matrix = linkage(sim_matrix, "ward")

                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, n_sampled, weights
                )

            sampled_clients = sample_clients(distri_clusters)

            # 对参与下一轮训练的客户端在其性能波动区间内随机抽取一个值作为实际能完成的任务量
            train_clients = []
            for j in sampled_clients:
                tasks_real = np.random.randint(clients_task_interval[j][0], clients_task_interval[j][1], 1)[0]
                if tasks_real < n_samples[j]:
                    # 真实能够完成的任务量小于样本量的，该客户端掉队
                    predict_flag[j][0] = predict_flag[j][0] + 1
                else:
                    # 样本量小于等于真实能够完成的，表明该客户端顺利完成训练任务
                    predict_flag[j][1] = predict_flag[j][1] + 1
                    train_clients.append(j)

            train_clients = np.array(train_clients)

            # 选择出来的客户端轮流开始训练
            # 选择出来的客户端轮流开始训练
            for index, k in enumerate(train_clients):
                # 相当于服务器下发模型
                local_model = deepcopy(model)
                # 设置本地客户端的优化器
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
                # 客户端k开始训练
                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

            # CREATE THE NEW GLOBAL MODEL AND SAVE IT
            if len(train_clients) > 0:
                model = FedAvg_agregation_process(
                    deepcopy(model), clients_params, weights=[1 / len(train_clients)] * len(train_clients)
                )

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        if i % metric_period == 0:

            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # UPDATE THE HISTORY OF LATEST GRADIENT
        if sampling == "S_clustered_2":
            gradients_i = get_gradients(
                sampling, previous_global_model, clients_models
            )
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
                gradients[idx] = gradient

        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def simulate_FedAvg_sampling(
        model,
        n_sampled,
        training_sets: list,
        testing_sets: list,
        n_iter: int,
        n_SGD: int,
        lr,
        file_name: str,
        decay=1,
        metric_period=1,
        mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # 每个客户端的性能波动区间，即一个epoch时间内完成的任务量波动区间
    clients_task_interval = [[150, 200] for _ in range(10)] + \
                            [[250, 300] for _ in range(30)] + \
                            [[450, 550] for _ in range(30)] + \
                            [[600, 700] for _ in range(20)] + \
                            [[1000, 1100] for _ in range(10)]
    tasks_predict = n_samples
    predict_flag = np.zeros((100, 2))

    for i in range(n_iter):

        clients_params = []

        # 纯随机选择客户端 FedAVG原生方法
        # np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)
        print("sampled clients", sampled_clients)

        # 对参与下一轮训练的客户端在其性能波动区间内随机抽取一个值作为实际能完成的任务量
        train_clients = []
        for j in sampled_clients:
            tasks_real = np.random.randint(clients_task_interval[j][0], clients_task_interval[j][1], 1)[0]
            if tasks_real < n_samples[j]:
                # 真实能够完成的任务量小于样本量的，该客户端掉队
                predict_flag[j][0] = predict_flag[j][0] + 1
            else:
                # 样本量小于等于真实能够完成的，表明该客户端顺利完成训练任务
                predict_flag[j][1] = predict_flag[j][1] + 1
                train_clients.append(j)

        train_clients = np.array(train_clients)

        # 选择出来的客户端轮流开始训练
        for index, k in enumerate(train_clients):
            # 相当于服务器下发模型
            local_model = deepcopy(model)
            # 设置本地客户端的优化器
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            # 客户端k开始训练
            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process_for_FA_sampling(
            deepcopy(model),
            clients_params,
            weights=[weights[client] for client in train_clients],
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist
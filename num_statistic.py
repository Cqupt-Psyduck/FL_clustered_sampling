import pickle as pkl


def statistic_ucb_mbut():
    path_ucb_mbut = f"saved_exp_info/sampled_clients/" \
                    f"CIFAR10_nbal_1_S_UCB_MBUT2-10-0_any_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pkl"
    sampled_ucb_mbut = pkl.load(open(path_ucb_mbut, "rb"))
    num_ucb_mbut = 0
    for term in sampled_ucb_mbut:
        for i in term:
            if i == 1:
                num_ucb_mbut = num_ucb_mbut + 1
    return num_ucb_mbut


def statistic_lvir():
    path_lvir = f"saved_exp_info/sampled_clients/" \
                f"CIFAR10_nbal_1_S_clustered_2_cosine_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pkl"
    sampled_lvir = pkl.load(open(path_lvir, "rb"))
    num_lvir = 0
    for term in sampled_lvir:
        for i in term:
            if i == 1:
                num_lvir = num_lvir + 1
    return num_lvir


def statistic_rbcs():
    path_rbcs = f"saved_exp_info/sampled_clients/" \
                f"CIFAR10_nbal_1_S_clustered_1_any_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pkl"
    sampled_rbcs = pkl.load(open(path_rbcs, "rb"))
    num_rbcs = 0
    for term in sampled_rbcs:
        for i in term:
            if i == 1:
                num_rbcs = num_rbcs + 1
    return num_rbcs


def statistic_mdcs():
    path_mdcs = f"saved_exp_info/sampled_clients/" \
                f"CIFAR10_nbal_1_S_random_any_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pkl"
    sampled_mdcs = pkl.load(open(path_mdcs, "rb"))
    num_mdcs = 0
    for term in sampled_mdcs:
        for i in term:
            if i == 1:
                num_mdcs = num_mdcs + 1
    return num_mdcs


def statistic_random():
    path_random = f"saved_exp_info/sampled_clients/" \
                f"CIFAR10_nbal_1_S_FedAvg_any_i1000_N100_lr0.05_B50_d1.0_p0.1_m5_0.pkl"
    sampled_random = pkl.load(open(path_random, "rb"))
    num_random = 0
    for term in sampled_random:
        for i in term:
            if i == 1:
                num_random = num_random + 1
    return num_random


if __name__ == '__main__':
    print("UCB_MBUT", statistic_ucb_mbut())
    print("LVIR", statistic_lvir())
    print("RBCS", statistic_rbcs())
    print("MDCS", statistic_mdcs())
    print("Random", statistic_random())

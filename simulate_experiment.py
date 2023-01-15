#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

"""UPLOADING THE DATASETS"""
import sys

import torch

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(
    "dataset - sampling - sim_type - seed - n_SGD - lr - decay - p - force - mu"
)
# sys.argv[0]是文件名称
print(sys.argv[1:])

dataset = sys.argv[1]
sampling_non = sys.argv[2]
sim_type = sys.argv[3]
seed = int(sys.argv[4])
n_SGD = int(sys.argv[5])
lr = float(sys.argv[6])
decay = float(sys.argv[7])
p = float(sys.argv[8])
force = sys.argv[9] == "True"

try:
    mu = float(sys.argv[10])
except:
    mu = 0.0

"""GET THE HYPERPARAMETERS"""
from py_func.hyperparams import get_hyperparams

n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)
print("number of iterations", n_iter)
print("batch size", batch_size)
print("percentage of sampled clients", p)
print("metric_period", meas_perf_period)
print("regularization term", mu)

"""NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
from py_func.hyperparams import get_file_name

"""GET THE DATASETS USED FOR THE FL TRAINING"""
from py_func.read_db import get_dataloaders

list_dls_train, list_dls_test = get_dataloaders(dataset, batch_size)

"""NUMBER OF SAMPLED CLIENTS"""
n_sampled = int(p * len(list_dls_train))
print("number fo sampled clients", n_sampled)

"""LOAD THE INTIAL GLOBAL MODEL"""
from py_func.create_model import load_model

model_0 = load_model(dataset, seed).to(device)
print(model_0)

"""simulate with random sampling"""


def random(sampling: str):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    from py_func.UCB_MBUT_experiment import simulate_sampling_random

    simulate_sampling_random(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        meas_perf_period,
        mu,
    )


"""Run simulate with clustered sampling"""


def cluster(sampling: str, sim_type: str):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    from py_func.UCB_MBUT_experiment import simulate_clustered_sampling

    simulate_clustered_sampling(
        sampling,
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        sim_type,
        0,
        decay,
        meas_perf_period,
        mu,
    )

"""RUN simulate with its original sampling scheme sampling clients uniformly"""


def fedavg(sampling: str):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    from py_func.UCB_MBUT_experiment import simulate_FedAvg_sampling

    simulate_FedAvg_sampling(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        meas_perf_period,
        mu,
    )

"""simulate with MBUT-UCB sampling"""


def ucb_mbut(sampling: str):
    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)
    config = sampling[10:].split("-")
    bal_num = int(config[0])
    cluster_num = int(config[1])
    sigma = float(config[2])
    print(bal_num, cluster_num, sigma)
    from py_func.UCB_MBUT_experiment import simulate_UCB_MBUT_sampling
    from py_func.ClassDistribution import get_auxiliary_data_loader

    aux_data = get_auxiliary_data_loader(dataset)
    simulate_UCB_MBUT_sampling(
        model_0,
        n_sampled,
        list_dls_train,
        list_dls_test,
        aux_data,
        bal_num,
        cluster_num,
        sigma,
        n_iter,
        n_SGD,
        lr,
        file_name,
        decay,
        meas_perf_period,
        mu,
    )


if __name__ == '__main__':
    ucb_mbut("S_UCB_MBUT2-10-0")
    # cluster("S_clustered_2", "cosine")
    # cluster("S_clustered_1", "any")
    # random("S_random")
    # fedavg("S_FedAvg")
    print("done")

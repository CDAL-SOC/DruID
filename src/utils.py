import numpy as np

from sklearn.metrics import ndcg_score

import torch


k = 256
perf_k = 10

is_gpu = torch.cuda.is_available()
# is_gpu = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def nan2zero(X):
    return torch.where(torch.isnan(X), torch.zeros_like(X), X)


def nan2inf(X):
    return torch.where(torch.isnan(X), torch.zeros_like(X) + np.inf, X)


def get_zinb_loss(X_true, X_pred, X_theta, X_pi, is_mean, eps, ridge_lambda):
    # Based on https://www.nature.com/articles/s41467-018-07931-2
    # and https://github.com/theislab/dca/blob/master/dca/loss.py
    if is_gpu:
        ridge_lambda = torch.tensor([ridge_lambda]).cuda(device)
    else:
        ridge_lambda = torch.tensor([ridge_lambda])
    t1 = (
        torch.lgamma(X_theta + eps)
        + torch.lgamma(X_true + 1.0)
        - torch.lgamma(X_true + X_theta + eps)
    )
    t2 = (X_theta + X_true) * torch.log(1.0 + (X_pred / (X_theta + eps))) + (
        X_true * (torch.log(X_theta + eps) - torch.log(X_pred + eps))
    )
    nb_case = nan2inf(t1 + t2) - torch.log(1.0 - X_pi + eps)
    zero_nb = torch.pow(X_theta / (X_theta + X_pred + eps), X_theta)
    zero_case = -torch.log(X_pi + ((1.0 - X_pi) * zero_nb) + eps)
    result = torch.where(X_true < eps, zero_case, nb_case)
    ridge = ridge_lambda * (X_pi ** 2)
    result += ridge
    result = nan2inf(result)
    
    # if torch.sum(X_mu) == float('inf') or torch.isnan(torch.sum(X_mu)) or \
    #     torch.sum(result) == float('inf') or torch.isnan(torch.sum(result))
    #     #or \
    #     #(epoch%100 == 0):
    #     print("X_true: ")
    #     print(X_true)
    #     print("#")
    #     print("X_pred: ")
    #     print(X_pred)
    #     print("#")
    #     print("X_theta: ")
    #     print(X_theta)
    #     print("#")
    #     print("t1: ",t1)
    #     print("#")
    #     print("t2 (X_theta+X_true): ",(X_theta+X_true))
    #     print("t2 torch.log(1.0 + (X_pred/(X_theta+eps))) : ",torch.log(1.0 + (X_pred/(X_theta+eps))))
    #     print("t2 torch.log(X_theta+eps) : ",torch.log(X_theta+eps))
    #     print("t2 torch.log(X_pred+eps) : ",t2)
    #     print("t2 (X_true * (torch.log(X_theta+eps) - torch.log(X_pred+eps))) : ",(X_true * (torch.log(X_theta+eps) - torch.log(X_pred+eps))))
    #     print("#")
    #     print("self.__nan2inf(t1+t2): ")
    #     print(__nan2inf(t1+t2))
    #     print("#")
    #     print("X_pi: ")
    #     print(X_pi)
    #     print("#")
    #     print("torch.log(1.0-X_pi+eps): ")
    #     print(torch.log(1.0-X_pi+eps))
    #     print("#")
    #     print("nb_case: ",nb_case)
    #     print("#")
    #     print("zero_nb: ",zero_nb)
    #     print("#")
    #     print("zero_case: ",zero_case)
    #     print("#")
    #     print("result: ",result)
    #     print("#")
    #     print("ridge: ",ridge)
    #     print("#")
    #     print("result + ridge: ",result)
    #     print("#")
    #     print("torch.mean(result): ",torch.mean(result))
    #     print("torch.sum(result): ",torch.sum(result))
    #     print("#######")

    if is_mean:
        return torch.mean(result)
    else:
        return torch.sum(result)


def get_kld_loss(mu, logvar, dim=1, is_mean=False):
    # from https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim)
    #
    if is_mean:
        KLD = torch.mean(KLD * -0.5)
    else:
        KLD = torch.sum(KLD) * -0.5
    return KLD


def is_converged(prev_cost, cost, convg_thres):
    # diff = (prev_cost - cost)
    diff = abs(prev_cost) - abs(cost)
    if (abs(diff)) < convg_thres:
        return True
    if np.isnan(cost):
        return True

def get_zinorm_loss(X_true, X_pred, X_theta, X_pi, is_mean, eps, ridge_lambda):
    # Based on https://github.com/ajayago/NCMF_bioinformatics/blob/main/NCMF/src/loss.py
    if is_gpu:
        ridge_lambda = torch.tensor([ridge_lambda]).cuda(device)
    else:
        ridge_lambda = torch.tensor([ridge_lambda])
    constant_pi = torch.acos(torch.zeros(1)).item() * 2
    t1 = -torch.log(1.0 - X_pi)
    t2 = -0.5 * torch.log(2.0 * constant_pi * X_theta) - torch.square(X_true - X_pred) / ((2 * X_theta) + eps)
    norm_case = nan2inf(t1 - t2) #- torch.log(1.0 - X_pi + eps)
    zero_norm = 1.0 / torch.sqrt(2.0 * X_pi * X_theta + eps) * torch.exp(-0.5 * ((0. - X_pred) ** 2) / X_theta + eps)
    zero_case = -torch.log(X_pi + ((1.0 - X_pi) * zero_norm) + eps)
    result = torch.where(X_true < eps, zero_case, norm_case)
    ridge = ridge_lambda * (X_pi ** 2)
    result += ridge
    result = nan2inf(result)
    
    # if torch.sum(X_mu) == float('inf') or torch.isnan(torch.sum(X_mu)) or \
    #     torch.sum(result) == float('inf') or torch.isnan(torch.sum(result))
    #     #or \
    #     #(epoch%100 == 0):
    #     print("X_true: ")
    #     print(X_true)
    #     print("#")
    #     print("X_pred: ")
    #     print(X_pred)
    #     print("#")
    #     print("X_theta: ")
    #     print(X_theta)
    #     print("#")
    #     print("t1: ",t1)
    #     print("#")
    #     print("t2 (X_theta+X_true): ",(X_theta+X_true))
    #     print("t2 torch.log(1.0 + (X_pred/(X_theta+eps))) : ",torch.log(1.0 + (X_pred/(X_theta+eps))))
    #     print("t2 torch.log(X_theta+eps) : ",torch.log(X_theta+eps))
    #     print("t2 torch.log(X_pred+eps) : ",t2)
    #     print("t2 (X_true * (torch.log(X_theta+eps) - torch.log(X_pred+eps))) : ",(X_true * (torch.log(X_theta+eps) - torch.log(X_pred+eps))))
    #     print("#")
    #     print("self.__nan2inf(t1+t2): ")
    #     print(__nan2inf(t1+t2))
    #     print("#")
    #     print("X_pi: ")
    #     print(X_pi)
    #     print("#")
    #     print("torch.log(1.0-X_pi+eps): ")
    #     print(torch.log(1.0-X_pi+eps))
    #     print("#")
    #     print("nb_case: ",nb_case)
    #     print("#")
    #     print("zero_nb: ",zero_nb)
    #     print("#")
    #     print("zero_case: ",zero_case)
    #     print("#")
    #     print("result: ",result)
    #     print("#")
    #     print("ridge: ",ridge)
    #     print("#")
    #     print("result + ridge: ",result)
    #     print("#")
    #     print("torch.mean(result): ",torch.mean(result))
    #     print("torch.sum(result): ",torch.sum(result))
    #     print("#######")

    if is_mean:
        return torch.mean(result)
    else:
        return torch.sum(result)

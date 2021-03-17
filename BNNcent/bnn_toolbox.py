# February 17th, 2021
# Reproduction - Weight Uncertainty in Neural Networks
# Based on https://github.com/saxena-mayur/Weight-Uncertainty-in-Neural-Networks/tree/

# Import relevant package
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)

def log_gaussian_rho(x, mu, rho):
    return float(-0.5 * np.log(2 * np.pi)) - rho - (x - mu) ** 2 / (2 * torch.exp(rho) ** 2)

def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return GAUSSIAN_SCALER / sigma * bell

def mixture_prior(input, pi, s1, s2):
    prob1 = pi * gaussian(input, 0., s1)
    prob2 = (1. - pi) * gaussian(input, 0., s2)
    return torch.log(prob1 + prob2)

def ELBO(l_pw, l_qw, l_likelihood, beta):
    kl = beta * (l_qw - l_pw)
    return kl - l_likelihood

def probs(model, hyper, data, target):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(hyper.n_samples):
        output = torch.log(model(data))

        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.nll_loss(output, target, reduction='sum') * hyper.multiplier

        s_log_pw += sample_log_pw / hyper.n_samples
        s_log_qw += sample_log_qw / hyper.n_samples
        s_log_likelihood += sample_log_likelihood / hyper.n_samples

    return s_log_pw, s_log_qw, s_log_likelihood

def train(model, optimizer, loader, hyper, train=True):
    loss_sum = 0
    kl_sum = 0
    m = len(loader)

    for batch_id, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()

        beta = 1 / (m)

        l_pw, l_qw, l_likelihood = probs(model, hyper, data, target)
        loss = ELBO(l_pw, l_qw, l_likelihood, beta)
        loss_sum += loss / len(loader)

        if train:
            loss.backward()
            optimizer.step()
        else:
            kl_sum += (1. / len(loader)) * (l_qw - l_pw)
    if train:
        return loss_sum
    else:
        return kl_sum

def evaluate(model, loader, infer=True, samples=1):
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()

        if samples == 1:
            output = model(data, infer=infer)
        else:
            output = model(data)
            for i in range(samples - 1):
                output += model(data)

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return acc_sum / len(loader)


class HyperparametersInitialization(object):

    def __init__(self, hidden_units, lr=1e-5, mixture=True, max_epoch=600, batch_size=128, eval_batch_size=1000, n_samples=1, n_test_samples=10, pi=0.25, sigma_1=1, sigma_2=8):

        self.lr = lr
        self.hidden_units = hidden_units
        self.mixture = mixture
        self.max_epoch = max_epoch
        self.n_samples = n_samples
        self.n_test_samples = n_samples
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        
        self.pi = pi
        self.s1 = float(np.exp(-sigma_1))
        self.s2 = float(np.exp(-sigma_2))

        self.rho_init = -8 
        self.multiplier = 1. 
        self.momentum = 0.95 


class BNNLayer(nn.Module):
    def __init__(self, n_input, n_output, hyperparameters):
        super(BNNLayer, self).__init__()
        self.hyperparameters = hyperparameters
        self.n_input = n_input
        self.n_output = n_output

        self.s1 = hyperparameters.s1
        self.s2 = hyperparameters.s2
        self.pi = hyperparameters.pi

        # We initialise weigth_mu and bias_mu as for usual Linear layers in PyTorch
        self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input))
        self.bias_mu = nn.Parameter(torch.Tensor(n_output))

        torch.nn.init.kaiming_uniform_(self.weight_mu, nonlinearity='relu') # Fills the input Tensor with values according to the method described in “Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification” - He, K. et al. (2015), using a uniform distribution.
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_mu, -bound, bound)

        self.bias_rho = nn.Parameter(torch.Tensor(n_output).normal_(hyperparameters.rho_init, .05))
        self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(hyperparameters.rho_init, .05))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        if infer:
            output = F.linear(data, self.weight_mu, self.bias_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, 1).cuda())
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).cuda())
        W = self.weight_mu + torch.log(1+torch.exp(self.weight_rho)) * epsilon_W
        b = self.bias_mu + torch.log(1+torch.exp(self.bias_rho)) * epsilon_b

        output = F.linear(data, W, b)

        self.lqw = log_gaussian_rho(W, self.weight_mu, self.weight_rho).sum() + \
                   log_gaussian_rho(b, self.bias_mu, self.bias_rho).sum()

        if self.hyperparameters.mixture:
            self.lpw = mixture_prior(W, self.pi, self.s2, self.s1).sum() + \
                       mixture_prior(b, self.pi, self.s2, self.s1).sum()
        else:
            self.lpw = log_gaussian(W, 0, self.s1).sum() + log_gaussian(b, 0, self.s1).sum()

        return output


class BNN(nn.Module):
    def __init__(self, n_input, n_ouput, hyperparameters):
        super(BNN, self).__init__()
        self.n_input = n_input
        self.layers = nn.ModuleList([])

        print('[INFO] Weight initialization...')
        self.layers.append(BNNLayer(n_input, hyperparameters.hidden_units, hyperparameters))
        self.layers.append(BNNLayer(hyperparameters.hidden_units, hyperparameters.hidden_units, hyperparameters))
        self.layers.append(BNNLayer(hyperparameters.hidden_units, n_ouput, hyperparameters))

    def forward(self, data, infer=False):
        output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
        output = F.relu(self.layers[1](output, infer))
        output = F.softmax(self.layers[2](output, infer), dim=1)
        return output

    def get_lpw_lqw(self):
        lpw = self.layers[0].lpw + self.layers[1].lpw + self.layers[2].lpw
        lqw = self.layers[0].lqw + self.layers[1].lqw + self.layers[2].lqw
        return lpw, lqw
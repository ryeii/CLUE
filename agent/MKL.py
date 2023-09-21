import pytorch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gpytorch
import os
import pandas as pd

import data.data_manager as data_manager

'''Define GP Model class'''
class MyGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Mean Function
        self.mean_module = gpytorch.means.ZeroMean()
        # Kernel Function
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class MKL:

    def __init__(self) -> None:
        pass

    def meta_kernel_learning(X, Y, epochs=1, save_path='gp_model.pth', verbose=True):
        train_x_tensors = []
        train_y_tensors = []
        # segment train and test sets into 1000 rows each
        for i in range(0, X.shape[0]-1000, 1000):
            train_x_tensors.append(torch.tensor(X[i:i+1000].values, dtype=torch.float))
            train_y_tensors.append(torch.tensor(Y[i:i+1000].values, dtype=torch.float))
        # reshape all train_y_tensors
        train_y_tensors = [train_y_tensor.reshape(-1) for train_y_tensor in train_y_tensors]
        # for all train_x_tensors and train_y_tensors, train a gp kernel
        for i in range(len(train_x_tensors)):
            train_x = train_x_tensors[i]
            train_y = train_y_tensors[i]
            # initialize likelihood and model
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = MyGP(train_x, train_y, likelihood)
            if os.path.exists(save_path + '_temp'):
                model.load_state_dict(torch.load(save_path + '_temp'))
            # Put model in train mode
            model.train()
            likelihood.train()
            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Define loss function the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            # training step
            for _ in range(epochs):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.sum().backward()
                optimizer.step()
            # save model
            temperary_save_path = save_path + '_temp'
            torch.save(model.state_dict(), temperary_save_path)
            # print progress
            if verbose:
                print('Epoch: {}/{}...'.format(i+1, len(train_x_tensors)), 'Loss: {:.4f}'.format(loss.item()), 'Lengthscale: {:.4f}'.format(model.covar_module.base_kernel.lengthscale.item(), 'Outputscale: {:.4f}'.format(model.covar_module.outputscale.item())))
        # save model
        torch.save(model.state_dict(), save_path)
        del model, likelihood, optimizer, mll

    '''Helper function to construct meta-learning-set X and Y with data_manager'''
    def construct_meta_sets(city_, season_):
        X = pd.DataFrame()
        Y = pd.DataFrame()
        for city in data_manager.data_tree.keys():
            for season in data_manager.data_tree[city].keys():
                if city == city_ and season == season_:
                    continue
                else:
                    x_, y_ = data_manager.data_tree[city][season]
                    X = pd.concat([X, x_], axis=0)
                    Y = pd.concat([Y, y_], axis=0)
        return X, Y


'''Example usage'''
# os.makedirs('meta_gp_models') if not os.path.exists('meta_gp_models') else None
# # for experiment analysis purpose (to test stability of meta gp model)
# for i in range(5):
#     os.mkdir('meta_gp_models_{}'.format(i)) if not os.path.exists('meta_gp_models_{}'.format(i)) else None

mkl = MKL()
for city in data_manager.data_tree.keys():
    for season in data_manager.data_tree[city].keys():
        x_set, y_set = mkl.construct_meta_sets(city_=city, season_=season)
        mkl.meta_kernel_learning(X=x_set, Y=y_set, save_path='meta_gp_models_{}/gp_model_{}_{}.pth'.format(0, city, season))
        print('meta gp model trained for {} {}'.format(city, season))

import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch
import pandas as pd
import os

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

'''Fine tune GP model from a meta-learned model'''
def train_model(X_data, Y_data, model_path, verbose=True):
    # tensorize X_data and Y_data
    X_data = torch.tensor(X_data.values, dtype=torch.float)
    Y_data = torch.tensor(Y_data.values, dtype=torch.float)
    # reshape Y_data
    Y_data = Y_data.reshape(-1)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MyGP(X_data, Y_data, likelihood)
    # read model from model_path
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    training_iter = 200
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_data)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_data)
        loss.backward()
        if verbose:
            if i % 1 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
        optimizer.step()
    # finetuned model is saved in 'meta_gp_model_(temp).pth'
    torch.save(model.state_dict(), 'meta_gp_model_(temp).pth')
    
'''Helper function to predict y given x'''''
def gp_predict(x, gpmodel):
    # predict y
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    with torch.no_grad():
        observed_pred = gpmodel(x)
    return observed_pred.mean.numpy(), observed_pred.variance.numpy()

'''Run the finetuned model on the historical data and evaluate accuracy'''
def run_experiment(X, Y, sample_size, error_threshold, var_threshold, gpmodel, plot=False):
    # Here the test set is the 3000 rows after the sample_size,\
    # however, this test set can be changed to any other data.
    test_x = torch.tensor(X[sample_size:sample_size+3000].values, dtype=torch.float)
    test_y = Y[sample_size:sample_size+3000].values
    # reshape test_y
    test_y = test_y.reshape(-1)
    y_pred, y_var = gp_predict(test_x, gpmodel)
    absolute_error = np.abs(y_pred - test_y)
    
    '''Calculate statistics starts here'''
    # positive events is an array that stores time steps when absolute error is greater than or equal to error_threshold
    P = np.where(absolute_error >= error_threshold)[0]
    N = np.where(absolute_error < error_threshold)[0]
    # true positive events is an array that stores time steps when variance is greater than or equal to var_threshold AND absolute error is greater than or equal to error_threshold
    TP = np.where((y_var >= var_threshold) & (absolute_error >= error_threshold))[0]
    TN = np.where((y_var < var_threshold) & (absolute_error < error_threshold))[0]
    # false positive events is an array that stores time steps when variance is greater than or equal to var_threshold AND absolute error is less than error_threshold
    FP = np.where((y_var >= var_threshold) & (absolute_error < error_threshold))[0]
    # accuracy (ACC) = (TP + TN) / (P + N)
    ACC = (len(TP) + len(TN)) / (len(P) + len(N))
    # define edge cases to avoid division by zero
    if len(P) == 0:
        TPR = 0
    else:
        TPR = len(TP) / len(P)
    # define edge cases to avoid division by zero
    if len(TP) + len(FP) == 0:
        PVV = 0
    else:
        PVV = len(TP) / (len(TP) + len(FP))
    '''Calculate statistics ends here'''

    if plot:
        # plot predictions
        plt.figure(figsize=(20, 5))
        plt.plot(test_y, label='ground truth')
        plt.plot(y_pred, label='prediction')
        plt.fill_between(np.arange(7*96), y_pred-y_var, y_pred+y_var, alpha=0.5)
        plt.legend()
        plt.show()

    # return statistics
    return ACC, TPR, PVV


def get_results_for_sample_size(sample_size_):
    # create a dataframe with columns: city, season, sample_size, error_threshold, var_threshold, ACC, TPR, PVV
    results = pd.DataFrame(columns=['city', 'season', 'sample_size', 'error_threshold', \
                                    'var_threshold', 'ACC', 'TPR', 'PVV'])

    # Choose cities from ['port_angeles', 'pittsburgh', 'tucson', 'new_york']
    # Choose seasons from ['summer', 'winter', 'all_year']
    for city in ['pittsburgh']:
        for season in ['winter']:
            # fetch data
            x_, y_ = data_manager.data_tree[city][season]
            # fine tune model
            train_model(x_[:sample_size_], y_[:sample_size_], 
                        model_path='meta_gp_models_0/gp_model_{}_{}.pth'.format(city, season))
            train_x = torch.tensor(x_[:sample_size_].values, dtype=torch.float)
            train_y = torch.tensor(y_[:sample_size_].values, dtype=torch.float)
            # reshape train_y
            train_y = train_y.reshape(-1)
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # fit GP model
            model = MyGP(train_x=train_x, train_y=train_y, likelihood=likelihood)
            model.load_state_dict(torch.load('meta_gp_model_(temp).pth'))
            # Put model in eval mode for prediction
            model.eval()
            likelihood.eval()

            '''Run experiment with different error_threshold and var_threshold'''
            # specify any error_threshold to be tested
            for error_threshold in [.5, 1, 1.5, 2, 2.5, 3]:
                # for each error_threshold, iterate through a list of var_thresholds\
                # from 0.1 to error_threshold+0.6 with step size 0.1.\
                # the range of var_thresholds is chosen by heuristics.
                for var_threshold in np.arange(0.1, error_threshold+0.6, 0.1):
                    ACC, TPR, PVV = run_experiment(X=x_, Y=y_, sample_size=sample_size_, 
                                                   error_threshold=error_threshold, 
                                                   var_threshold=var_threshold, 
                                                   gpmodel=model, plot=False)
                    new_data = pd.DataFrame({'city': [city],
                            'season': [season],
                            'sample_size': [sample_size_],
                            'error_threshold': [error_threshold],
                            'var_threshold': [var_threshold],
                            'ACC': [ACC],
                            'TPR': [TPR],
                            'PVV': [PVV]})
                    results = pd.concat([results, new_data], ignore_index=True)
                    print('city: {}, season: {}, sample_size: {}, error_threshold: {},\
                           var_threshold: {}, ACC: {}, TPR: {}, PVV: {}'
                          .format(city, season, sample_size_, error_threshold, var_threshold, ACC, TPR, PVV))
    # save results
    results.to_csv('meta_gp_test_results_{}.csv'.format(sample_size_), index=False)

'''Translator function takes the test results and choose the \
    optimal var_threshold for each error_threshold'''
def translator(result_path, save_path):
    # read results
    results = pd.read_csv(result_path)
    # initialize a dataframe to store optimal var_thresholds
    optimal_var_thresholds = pd.DataFrame(columns=['city', 'season', 'sample_size', 'error_threshold', 'var_threshold', 'ACC', 'TPR', 'PVV'])
    # iterate through each error_threshold
    for error_threshold in results['error_threshold'].unique():
        # for each error_threshold, find the optimal var_threshold
        optimal_var_threshold = results[results['error_threshold'] == error_threshold].sort_values(by='ACC', ascending=False).iloc[0]
        # append the optimal var_threshold to optimal_var_thresholds
        optimal_var_thresholds = pd.concat([optimal_var_thresholds, optimal_var_threshold], ignore_index=True)
    # save optimal_var_thresholds
    optimal_var_thresholds.to_csv(save_path, index=False)


'''Example usage'''
sample_sizes = [2000]
for i in sample_sizes:
    get_results_for_sample_size(i)
    translator('meta_gp_test_results_{}.csv'.format(i), 'meta_gp_optimal_var_thresholds_{}.csv'.format(i))


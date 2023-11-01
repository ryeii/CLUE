import numpy as np
import pandas as pd
from datetime import datetime
import gymnasium as gym
import sinergym
import torch, torch.nn as nn
import gpytorch

import data.data_manager as data_manager

'''Define the GP model class'''
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

'''Helper function to predict the next state vector'''
def model_predict_fn(state_vector, model):
    # tensorize the state vector
    state_vector = torch.tensor(state_vector.astype(np.float32))
    with torch.no_grad():
        observed_pred = model(state_vector)
    return observed_pred.mean.numpy(), observed_pred.variance.numpy()

'''We illustrate CLUE by implementing confidence-based control with random-shooting method'''
def choose_action(state, model_predict_fn, horizon, n_samples, environment_forecast, curr_step, model, epsilon, winter=True):
    # initialize the zone temperature matrix
    zone_temperature_matrix = np.zeros((n_samples, horizon+1, 1))
    # populate every n_samples at time 0 with the current zone temperature
    for i in range(n_samples):
        zone_temperature_matrix[i, 0] = state
    # initialize the action matrix
    action_matrix = np.zeros((n_samples, horizon, 2))
    # populate the action matrix with random integers between 15 and 26 inclusive
    for i in range(n_samples):
        for j in range(horizon):
            action_matrix[i, j, 0] = np.random.randint(21, 30)
            action_matrix[i, j, 1] = np.random.randint(15, 21)
    # get the environment forecast for the next horizon steps
    environment_forecast = environment_forecast[curr_step:curr_step+horizon]
    # initialize the uncertainty matrix
    uncertainty_matrix = np.zeros((n_samples, horizon))
    # loop through the horizon
    for t in range(horizon):
        # get the action vector, zone temperature vector, and environment forecast vector at time t
        action_vector = action_matrix[:, t]
        zone_temperature_vector = zone_temperature_matrix[:, t]
        environment_forecast_vector = np.tile(environment_forecast.iloc[t], (n_samples, 1))
        # concatenate the three vectors to form the state vector as input to the GP model
        state_vector = np.concatenate([environment_forecast_vector, zone_temperature_vector, action_vector], axis=1)
        # make prediction using the GP model and get the next zone temperature vector and uncertainty vector
        next_zone_temperature_vector, uncertainty_vector = model_predict_fn(state_vector, model)
        # reshape the next_zone_temperature_vector to be (n_samples, 1)
        next_zone_temperature_vector = next_zone_temperature_vector.reshape((n_samples, 1))
        # reshape the uncertainty_vector to be (n_samples, 1)
        uncertainty_matrix[:, t] = uncertainty_vector
        # populate the zone_temperature_matrix with the next_zone_temperature_vector
        zone_temperature_matrix[:, t+1] = next_zone_temperature_vector
    # remove the first column of the uncertainty matrix
    zone_temperature_matrix = zone_temperature_matrix[:, 1:]
    # if all confidence value in the first column is larger than epsilon, then rule-based
    if np.sum(uncertainty_matrix[:, 0] > epsilon) == n_samples:
        # the rule-based action is to turn on the HVAC to the default setpoint (21, 21)\
        # if there is at least one person in the zone, otherwise turn off the HVAC
        if environment_forecast.iloc[0].at['Zone People Occupant Count(SPACE1-1)'] > 0:
            action = (21, 21)
        else:
            action = (15, 30)
        return action
    # discard the rows in the zone_temperature_matrix, action_matrix, and environment_forecast that\
    # have uncertainty value larger than epsilon
    zone_temperature_matrix = zone_temperature_matrix[uncertainty_matrix[:, 0] <= epsilon]
    action_matrix = action_matrix[uncertainty_matrix[:, 0] <= epsilon]
    environment_forecast = environment_forecast[uncertainty_matrix[:, 0] <= epsilon]
    uncertainty_matrix = uncertainty_matrix[uncertainty_matrix[:, 0] <= epsilon]
    # evaluate the rewards for each trajectory
    rewards_vector = evaluate_rewards(zone_temperature_matrix, action_matrix, environment_forecast, uncertainty_matrix, winter)
    # return the action that maximizes the rewards
    return action_matrix[np.argmax(rewards_vector), 0]

'''The reward function'''
def evaluate_rewards(zone_temperature_matrix, action_matrix, env_forecast, uncertainty_matrix, winter, gamma=0.99, lambda_=0.5):
    rewards_vector = np.zeros(zone_temperature_matrix.shape[0])
    for trajectory in range(zone_temperature_matrix.shape[0]):
        sum_reward = 0
        for t in range(zone_temperature_matrix.shape[1]):
            # get 'Zone People Occupant Count(SPACE1-1)' in the env_forcast
            people_count = env_forecast.iloc[t].at['Zone People Occupant Count(SPACE1-1)']
            zone_temp = zone_temperature_matrix[trajectory, t]
            action = action_matrix[trajectory, t]
            step_reward = 0
            if people_count > 0:
                # the comfort zone for summer is 23-26 degC, for winter is 20-23.5 degC
                if not winter:
                    if zone_temp < 23.0:
                        step_reward += zone_temp - 23.0
                    elif zone_temp > 26.0:
                        step_reward += 26.0 - zone_temp
                else:
                    if zone_temp < 20.0:
                        step_reward += zone_temp - 20.0
                    elif zone_temp > 23.5:
                        step_reward += 23.5 - zone_temp
            else:
                if not winter:
                    step_reward -= abs(action[1]-15)
                else:
                    step_reward -= abs(action[0]-30)
            # add the uncertainty value to the step reward
            step_reward += lambda_*uncertainty_matrix[trajectory, t]
            # discount the step reward
            sum_reward += gamma**t*step_reward
        if type(sum_reward) == np.ndarray:
            rewards_vector[trajectory] += sum_reward[0]
        elif type(sum_reward) == np.float64:
            rewards_vector[trajectory] += sum_reward
    return rewards_vector

'''The agent outputs a float setpoint, which cannot be used if the environment is discrete.\
    In this case, we use an action mapping to convert the continuous action to a discrete one.\
    Otherwise, we can just use the action as is.'''
def run_experiment(data_set_size, city, n_samples, horizon, path, epsilon, log=True, monitor_path=None, overhead_path=None, winter=True):

    new_action_mapping = {
        0: (15, 30),
        1: (16, 29),
        2: (17, 28),
        3: (18, 27),
        4: (19, 26),
        5: (20, 25),
        6: (21, 24),
        7: (22, 23),
        8: (22, 22),
        9: (21, 21)
    }
    env_forecast = data_manager.get_environment_forecast('pittsburgh', 'summer')

    if winter:
        extra_params={'timesteps_per_hour' : 4,
                'runperiod' : (1,1,1997,31,1,1997)}
        env_forecast = data_manager.get_environment_forecast('pittsburgh', 'winter')
    else:
        extra_params={'timesteps_per_hour' : 4,
                'runperiod' : (1,7,1997,31,7,1997)}

    # specify which environment to create (pittsburgh, tucson, new_york)
    if city == 'pittsburgh':
        env = gym.make('Eplus-demo-v1', config_params=extra_params, action_mapping=new_action_mapping)
    elif city == 'tucson':
        env = gym.make('Eplus-5Zone-hot-discrete-v1', config_params=extra_params, action_mapping=new_action_mapping)
    elif city == 'new_york':
        env = gym.make('Eplus-5Zone-mixed-discrete-v1', config_params=extra_params, action_mapping=new_action_mapping)

    obs, info = env.reset()
    terminated = False
    time_overhead = []

    current_step = 0

    monitor = pd.DataFrame(columns=['year','month','day','hour','Site Outdoor Air Drybulb Temperature(Environment)',
                                    'Site Outdoor Air Relative Humidity(Environment)','Site Wind Speed(Environment)',
                                    'Site Wind Direction(Environment)','Site Diffuse Solar Radiation Rate per Area(Environment)',
                                    'Site Direct Solar Radiation Rate per Area(Environment)','Zone Thermostat Heating Setpoint Temperature(SPACE1-1)',
                                    'Zone Thermostat Cooling Setpoint Temperature(SPACE1-1)','Zone Air Temperature(SPACE1-1)','Zone Air Relative Humidity(SPACE1-1)',
                                    'Zone People Occupant Count(SPACE1-1)','People Air Temperature(SPACE1-1 PEOPLE 1)','Facility Total HVAC Electricity Demand Rate(Whole Building)'])
    
    '''GP model fine-tuning starts here'''
    if winter:
        X_data, Y_data = data_manager.data_tree[city]['winter']
    else:
        X_data, Y_data = data_manager.data_tree[city]['summer']
    # one day is 96 timesteps
    X_data = X_data[:data_set_size*96]
    Y_data = Y_data[:data_set_size*96]
    # x and y data to numpy array
    X_data = X_data.to_numpy()
    Y_data = Y_data.to_numpy()
    # reshape Y_data
    Y_data = Y_data.reshape(-1)
    # convert to torch tensor
    X_data = torch.from_numpy(X_data).float()
    Y_data = torch.from_numpy(Y_data).float()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # limit the number of fitting data points to 700
    if len(X_data) > 700:
        model = MyGP(X_data[len(X_data)-700:], Y_data[len(X_data)-700:], likelihood)
    else:
        model = MyGP(X_data, Y_data, likelihood)
    model = MyGP(X_data, Y_data, likelihood)
    model.load_state_dict(torch.load(path))
    # train model
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(200):
        optimizer.zero_grad()
        output = model(X_data)
        loss = -mll(output, Y_data)
        loss.sum().backward()
        optimizer.step()
    '''GP model fine-tuning ends here'''

    # Put model in eval mode
    model.eval()
    likelihood.eval()
    while not terminated:
        t0 = datetime.now()
        if current_step == 0:
            obs = 0
        else:
            obs = obs['Zone Air Temperature(SPACE1-1)']
        # obs = obs[12]
        action = choose_action(obs, model_predict_fn, horizon, n_samples, env_forecast, curr_step=current_step, model=model, epsilon=epsilon, winter=winter)
        if winter:
            for i in new_action_mapping.keys():
                if new_action_mapping[i][1] == int(action[0]):
                    action = i
                    break
        else:
            for i in new_action_mapping.keys():
                if new_action_mapping[i][0] == int(action[1]):
                    action = i
                    break
        t1 = datetime.now()
        time_overhead.append(t1-t0)
        obs, reward, terminated, truncated, info = env.step(action)
        # add each item in the obs (a dictionary) to the monitor dataframe using concat
        monitor = pd.concat([monitor, pd.DataFrame(obs, index=[current_step])])
        current_step += 1
        if current_step % 100 == 0:
            print(current_step)
            monitor.to_csv(monitor_path)
    # turn time overhead into integer in milliseconds
    time_overhead = [int(t.total_seconds() * 1000) for t in time_overhead]
    # save time overhead to csv
    time_overhead_df = pd.DataFrame(time_overhead, columns=['time_overhead'])
    time_overhead_df.to_csv(overhead_path)
    # save monitor to csv 
    monitor.to_csv(monitor_path)
    env.close()


'''Example usage'''
days = 7
eps = .3
city = 'pittsburgh'
run_experiment(days, 
               city, 
               1000, 
               20, 
               path='./results/meta_gp_models/gp_model_{}_winter.pth'.format(city), 
               monitor_path='{}_GPMCB_winter_{}_{}_monitor.csv'.format(city, days, eps), 
               overhead_path='{}_GPMCB_winter_{}_{}_overhead.csv'.format(city, days, eps), 
               epsilon=eps, 
               winter=True)
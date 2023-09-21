import argparse
import json
import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import *
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor

import sinergym
import sinergym.utils.gcloud as gcloud
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.logger import CSVLogger, WandBOutputFormat
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *

# ---------------------------------------------------------------------------- #
#                             Parameters definition                            #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--configuration',
    '-conf',
    required=True,
    type=str,
    dest='configuration',
    help='Path to experiment configuration (JSON file)'
)
args = parser.parse_args()
# ------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------- #
#                             Read json parameters                             #
# ---------------------------------------------------------------------------- #

with open(args.configuration) as json_conf:
    conf = json.load(json_conf)

# ---------------------------------------------------------------------------- #
#                               Register run name                              #
# ---------------------------------------------------------------------------- #
experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
experiment_name = conf['algorithm']['name'] + '-' + conf['environment'] + \
    '-episodes-' + str(conf['episodes'])
if conf.get('seed'):
    experiment_name += '-seed-' + str(conf['seed'])
if conf.get('id'):
    experiment_name += '-id-' + str(conf['id'])
experiment_name += '_' + experiment_date

# ---------------------------------------------------------------------------- #
#                              WandB registration                              #
# ---------------------------------------------------------------------------- #

if conf.get('wandb'):
    # Create wandb.config object in order to log all experiment params
    experiment_params = {
        'sinergym-version': sinergym.__version__,
        'python-version': sys.version
    }
    experiment_params.update(conf)

    # Get wandb init params
    wandb_params = conf['wandb']['init_params']
    # Init wandb entry
    run = wandb.init(
        name=experiment_name + '_' + wandb.util.generate_id(),
        config=experiment_params,
        ** wandb_params
    )

# --------------------- Overwrite environment parameters --------------------- #
env_params = {}
# Transform required str's into Callables
if conf.get('env_params'):
    if conf['env_params'].get('reward'):
        conf['env_params']['reward'] = eval(conf['env_params']['reward'])
    if conf['env_params'].get('observation_space'):
        conf['env_params']['observation_space'] = eval(
            conf['env_params']['observation_space'])
    if conf['env_params'].get('action_space'):
        conf['env_params']['action_space'] = eval(
            conf['env_params']['action_space'])
    if conf['env_params'].get('action_mapping'):
        for key in list(conf['env_params']['action_mapping'].keys()):
            conf['env_params']['action_mapping'][int(
                key)] = conf['env_params']['action_mapping'].pop(key)

    env_params = conf['env_params']

# ---------------------------------------------------------------------------- #
#                           Environment construction                           #
# ---------------------------------------------------------------------------- #
# For this script, the execution name will be updated
env_params.update({'env_name': experiment_name})
env = gym.make(
    conf['environment'],
    ** env_params)
env = Monitor(env)

# env for evaluation if is enabled
eval_env = None
if conf.get('evaluation'):
    eval_name = conf['evaluation'].get('name', env.name + '-EVAL')
    env_params.update({'env_name': eval_name})
    eval_env = gym.make(
        conf['environment'],
        ** env_params)
    eval_env = Monitor(eval_env)

# ---------------------------------------------------------------------------- #
#                                   Wrappers                                   #
# ---------------------------------------------------------------------------- #
if conf.get('wrappers'):
    for key, parameters in conf['wrappers'].items():
        wrapper_class = eval(key)
        for name, value in parameters.items():
            # parse str parameters to sinergym Callable or Objects if it is
            # required
            if isinstance(value, str):
                if 'sinergym.' in value:
                    parameters[name] = eval(value)
        env = wrapper_class(env=env, ** parameters)
        if eval_env is not None:
            eval_env = wrapper_class(env=eval_env, ** parameters)

# ---------------------------------------------------------------------------- #
#                           Defining model (algorithm)                         #
# ---------------------------------------------------------------------------- #
algorithm_name = conf['algorithm']['name']
algorithm_parameters = conf['algorithm']['parameters']
if conf.get('model') is None:

    # --------------------------------------------------------#
    #                           DQN                          #
    # --------------------------------------------------------#
    if algorithm_name == 'SB3-DQN':

        model = DQN(env=env,
                    seed=conf.get('seed', None),
                    ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           DDPG                         #
    # --------------------------------------------------------#
    elif algorithm_name == 'SB3-DDPG':
        model = DDPG(env=env,
                     seed=conf.get('seed', None),
                     ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           A2C                          #
    # --------------------------------------------------------#
    elif algorithm_name == 'SB3-A2C':
        model = A2C(env=env,
                    seed=conf.get('seed', None),
                    ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           PPO                          #
    # --------------------------------------------------------#
    elif algorithm_name == 'SB3-PPO':
        model = PPO(env=env,
                    seed=conf.get('seed', None),
                    ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           SAC                          #
    # --------------------------------------------------------#
    elif algorithm_name == 'SB3-SAC':
        model = SAC(env=env,
                    seed=conf.get('seed', None),
                    ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           TD3                          #
    # --------------------------------------------------------#
    elif algorithm_name == 'SB3-TD3':
        model = TD3(env=env,
                    seed=conf.get('seed', None),
                    ** algorithm_parameters)
    # --------------------------------------------------------#
    #                           Error                        #
    # --------------------------------------------------------#
    else:
        raise RuntimeError(
            F'Algorithm specified [{algorithm_name} ] is not registered.')

else:
    model_path = ''
    if 'gs://' in conf['model']:
        # Download from given bucket (gcloud configured with privileges)
        client = gcloud.init_storage_client()
        bucket_name = conf['model'].split('/')[2]
        model_path = conf['model'].split(bucket_name + '/')[-1]
        gcloud.read_from_bucket(client, bucket_name, model_path)
        model_path = './' + model_path
    else:
        model_path = conf['model']

    model = None
    if algorithm_name == 'SB3-DQN':
        model = DQN.load(
            model_path)
    elif algorithm_name == 'SB3-DDPG':
        model = DDPG.load(
            model_path)
    elif algorithm_name == 'SB3-A2C':
        model = A2C.load(
            model_path)
    elif algorithm_name == 'SB3-PPO':
        model = PPO.load(
            model_path)
    elif algorithm_name == 'SB3-SAC':
        model = SAC.load(
            model_path)
    elif algorithm_name == 'SB3-TD3':
        model = TD3.load(
            model_path)
    else:
        raise RuntimeError('Algorithm specified is not registered.')

    model.set_env(env)

# ---------------------------------------------------------------------------- #
#       Calculating total training timesteps based on number of episodes       #
# ---------------------------------------------------------------------------- #
n_timesteps_episode = env.simulator._eplus_one_epi_len / \
    env.simulator._eplus_run_stepsize
timesteps = conf['episodes'] * n_timesteps_episode - 1

# ---------------------------------------------------------------------------- #
#                                   CALLBACKS                                  #
# ---------------------------------------------------------------------------- #
callbacks = []

# Set up Evaluation and saving best model
if conf.get('evaluation'):
    eval_callback = LoggerEvalCallback(
        eval_env,
        best_model_save_path=eval_env.simulator._env_working_dir_parent +
        '/best_model/',
        log_path=eval_env.simulator._env_working_dir_parent +
        '/best_model/',
        eval_freq=n_timesteps_episode *
        conf['evaluation']['eval_freq'],
        deterministic=True,
        render=False,
        n_eval_episodes=conf['evaluation']['eval_length'])
    callbacks.append(eval_callback)

# Set up wandb logger
if conf.get('wandb'):
    # wandb logger and setting in SB3
    logger = Logger(
        folder=None,
        output_formats=[
            HumanOutputFormat(
                sys.stdout,
                max_length=120),
            WandBOutputFormat()])
    model.set_logger(logger)
    # Append callback
    log_callback = LoggerCallback()
    callbacks.append(log_callback)


callback = CallbackList(callbacks)

# ---------------------------------------------------------------------------- #
#                                   TRAINING                                   #
# ---------------------------------------------------------------------------- #
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=conf['algorithm']['log_interval'])
model.save(env.simulator._env_working_dir_parent + '/model')

# If the algorithm doesn't reset or close the environment, this script will do it in
# order to correctly log all the simulation data (Energyplus + Sinergym
# logs)
if env.simulator._episode_existed:
    env.close()

# ---------------------------------------------------------------------------- #
#                              Wandb artifact log                              #
# ---------------------------------------------------------------------------- #

if conf.get('wandb'):
    artifact = wandb.Artifact(
        name=conf['wandb']['artifact_name'],
        type=conf['wandb']['artifact_type'])
    artifact.add_dir(
        env.simulator._env_working_dir_parent,
        name='training_output/')
    if conf.get('evaluation'):
        artifact.add_dir(
            eval_env.simulator._env_working_dir_parent,
            name='evaluation_output/')
    run.log_artifact(artifact)

# wandb has finished
run.finish()

# ---------------------------------------------------------------------------- #
#                      Google Cloud Bucket Storage                             #
# ---------------------------------------------------------------------------- #
if conf.get('cloud'):
    if conf['cloud'].get('remote_store'):
        # Initiate Google Cloud client
        client = gcloud.init_storage_client()
        # Code for send output to common Google Cloud resource here.
        gcloud.upload_to_bucket(
            client,
            src_path=env.simulator._env_working_dir_parent,
            dest_bucket_name=conf['cloud']['remote_store'],
            dest_path=name)
        if conf.get('evaluation'):
            gcloud.upload_to_bucket(
                client,
                src_path='best_model/' + name + '/',
                dest_bucket_name=conf['cloud']['remote_store'],
                dest_path='best_model/' + name + '/')
    # ---------------------------------------------------------------------------- #
    #                   Autodelete option if is a cloud resource                   #
    # ---------------------------------------------------------------------------- #
    if conf['cloud'].get(
            'remote_store') and conf['cloud'].get('auto_delete'):
        token = gcloud.get_service_account_token()
        gcloud.delete_instance_MIG_from_container(
            conf['cloud']['group_name'], token)

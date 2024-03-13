"""
This is a boilerplate pipeline 'rnd_dqn'
generated using Kedro 0.19.1
"""

import d3rlpy
import gym 
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, InfiniteBuffer
from d3rlpy.algos import DQNConfig
from d3rlpy.metrics import EnvironmentEvaluator
import numpy as np

def random_collector(params):
    n_steps = params["collect_steps"]
    #Make environment 
    env = gym.make("CartPole-v1")
    #Random policy provided by d3rlpy
    random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)
    # start data collection
    random_policy.collect(env, buffer, n_steps=n_steps)

    #print("DS-EXPORT",buffer.transition_picker(episode=buffer.episodes[0], index=2))
    return buffer

def dqn_training(dataset, params):
    print("dataset_size", dataset.size())
    n_steps = params["collect_steps"]

    #Debug print
    #print("dataset info after",dataset.transition_picker(episode=dataset.episodes[0], index=2))

    #If you don't use GPU, set device=None instead.
    dqn = DQNConfig().create(device=None)
    dqn.build_with_dataset(dataset)

    #training TODO: understand how many training steps are appropriate and what one training step is.
    dqn.fit(dataset, n_steps = params["collect_steps"]
,)

    #Debug print
    #print("EXPORT:",dqn.impl.modules)
    return dqn

def online_evaluation(model, params):
    #print("IMPORT:",model.impl.modules)
    def obs_format(observation):
            if isinstance(observation, np.ndarray):
                return np.expand_dims(observation, axis=0)
            elif isinstance(observation, (tuple, list)):
                observation = [np.expand_dims(o, axis=0) for o in observation]
                return observation
            else:
                raise ValueError(f"Unsupported observation type: {type(observation)}")

    algo = model
    episode_rewards = []
    for j in range(params["test_eps"]):
            env = gym.make("CartPole-v1")
            observation, info = env.reset()
            terminated = False
            truncated = False
            cum_reward=0.0

            while not terminated and not truncated:
                form_obs = obs_format(observation)
                action = algo.predict(form_obs)[0]
                observation, reward, terminated, truncated, info = env.step(action)
                cum_reward += reward
                env.close()
            episode_rewards.append(cum_reward)
    mean_rew=float(np.mean(episode_rewards))
    std_rew=float(np.std(episode_rewards))

    print("Mean Reward:", mean_rew, "Mean standard deviation", std_rew, "for", j, "played Episodes")

    env = gym.make("CartPole-v1",  render_mode="human")
    observation, info = env.reset()
    env.render()
    terminated = False
    truncated = False
    cum_reward = 0
    while terminated == False and truncated == False:
        form_obs = obs_format(observation)
        action = algo.predict(form_obs)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
    env.close()
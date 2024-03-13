"""
This is a boilerplate pipeline 'tpg_dqn'
generated using Kedro 0.19.1
"""

import sys
sys.path.append(r'C:\Users\david\Dropbox\GitHub\PyTPG')
from tpg.trainer import Trainer # Import to train 
from tpg.agent import Agent # Import to run
import time 
import gym
import numpy as np
import d3rlpy
import numpy as np

class d3_data:
    states = []
    actions = []
    rewards = []
    term_list = [] 

    def save_data(self): 
        buffer=d3rlpy.dataset.MDPDataset(
                    observations=np.array(self.states),
                    actions=np.array(self.actions),
                    rewards=np.array(self.rewards),
                    terminals=np.array(self.term_list),
                )
        print("SIZE OF DATASET:", buffer.size())
        # with open(f"{name}.h5", "w+b") as f:
        #     buffer.dump(f)
        return buffer

def tpg_collector(params):
    #Hyperparameters
    env_str = params["env_str"]
    generations = params["generations"]
    sel_ep = params["sel_ep"]
    pop_size= params["pop_size"]

    #Data Collector
    coll = d3_data()

    #Gym and tome
    tStart = time.time()
    env = gym.make(env_str)

    #Population
    trainer = Trainer(actions=env.action_space.n, teamPopSize=pop_size) 

    curScores = [] # hold scores in a generation
    summaryScores = [] # record score summaries for each gen (min, max, avg)

    for gen in range(generations): # generation loop
        print("Generation", gen)
        curScores = [] # new list per gen
        agents = trainer.getAgents() #load the agents 
        
        while True: # loop to go through agents
            teamNum = len(agents)
            agent = agents.pop()
            # if agent is None:
            #     print("EMPTY")
            #     break # no more agents, so proceed to next gen
            
            # get initial state and prep environment
            cum_reward = 0
            observation, info = env.reset()
            terminated = False
            truncated = False

            for i in range(sel_ep): # run x episodes
                while terminated == False and truncated == False:
                    action = agent.act(observation) 
                    coll.states.append(observation)
                    coll.actions.append([action])
                    observation, reward, terminated, truncated, info = env.step(action)
                    coll.rewards.append([reward])
                    cum_reward += reward # accumulate reward in score
                    if terminated or truncated:
                        coll.term_list.append([1])
                    else:
                        coll.term_list.append([0])

            agent.reward(cum_reward) # must reward agent (if didn't already score)
            curScores.append(cum_reward) # store score -> what
            
            if len(agents) == 0:
                break
                
        # at end of generation, make summary of scores
        summaryScores.append((min(curScores), max(curScores),
                        sum(curScores)/len(curScores))) # min, max, avg
        trainer.evolve()

    #clear_output(wait=True)
    print('Time Taken (Hours): ' + str((time.time() - tStart)/3600))
    print('Results:\nMin, Max, Avg')
    for result in summaryScores:
        print(result[0],result[1],result[2])

    #Save the dataset
    return coll.save_data()


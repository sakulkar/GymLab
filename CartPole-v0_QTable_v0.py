from __future__ import division
import math
import gym
# from time import sleep

## Custom imports
from agents.QTable import QAgent
from agents.Quantizer import Quantizer

env = gym.make('CartPole-v0') # Create environment

######## Check Observation Space ########
# print(env.observation_space.shape)
# print(env.observation_space.high)
# print(env.observation_space.low)

################ Observation Quantization parameters #####################
buckets = [1, 1, 6, 3]
low = env.observation_space.low
high = env.observation_space.high

low[1] = -0.5
low[3] = -math.radians(50)
high[1] = 0.5
high[3] = math.radians(50)

# print low 
# print high
#############################################################################

###################### Utility functions ####################################
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1.0, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
#############################################################################

## Learning related constants
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
learning_rate = get_learning_rate(0)
explore_rate = get_explore_rate(0)

action_set = range(env.action_space.n) # Set of valid actions
agent = QAgent(actions=action_set, alpha=learning_rate, gamma=0.99, explore_strategy='epsilon', epsilon=explore_rate) # Create an agent
quantizer = Quantizer(low=low, high=high, buckets=buckets) # Create a observation quantizer

# simulation parameters
NUM_EPISODES = 1000     # Total number of episodes
MAX_STEPS = 250         # Max number of steps in one episode
MAX_T = 250             
STREAK_TO_END = 120     
SOLVED_T = 199          # Condition on solved decision
DEBUG_MODE = False      # Verbose: Prints intermediate values
ENABLE_UPLOAD = True

if ENABLE_UPLOAD:
    env.monitor.start('/tmp/cart_pole_q_table_exp_01')

num_streaks = 0 # Total number of streaks till now

for idx_episode in range(NUM_EPISODES):
    # Episode initialization
    observation = env.reset()
    reward = 1
    agent.reset(foget_table=False)

    # Set ALPHA: learning rate
    learning_rate = get_learning_rate(idx_episode)
    agent.update_learning_rate(learning_rate)

    # Set EPSILON: exploration
    explore_rate = get_explore_rate(idx_episode)
    agent.update_exploration_rate(explore_rate)

    # print agent.q_table

    for step_idx in range(MAX_STEPS):
        env.render()
        quant_obs = quantizer.quantize(observation)    # Quantize the observation
        action, _ = agent.observe_and_act(observation=quant_obs, last_reward=reward)  # get and random action
        ####action = env.action_space.sample() # Random Action
        observation, reward, done, info = env.step(action)
        
        # Print data
        if (DEBUG_MODE):
            print("\nEpisode = %d" % idx_episode)
            print("t = %d" % step_idx)
            print("Action: %d" % action)
            print("State: %s" % str(observation))
            print("Reward: %f" % reward)
            #print("Best Q: %f" % best_q)
            print("Explore rate: %f" % explore_rate)
            print("Learning rate: %f" % learning_rate)
            print("")

        if done or (step_idx == MAX_STEPS-1):
            print "Episode {} finished after {} timesteps".format(idx_episode+1, step_idx+1)
            if (step_idx >= SOLVED_T):
                num_streaks += 1
            else:
               num_streaks = 0           
            # sleep(0.1)
            break


    if num_streaks >= STREAK_TO_END:
        print "Problem Solved!"
        break

if ENABLE_UPLOAD:
    env.monitor.close()
    gym.upload('/tmp/cart_pole_q_table_exp_01', api_key='sk_5kpnfZ3lTnekeluRy3YtvQ')
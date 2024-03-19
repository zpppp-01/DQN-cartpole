# Born time: 2022-5-10
# Latest update: 2023-12-8
# RL Training Phase
# Dylan


import gym
import numpy as np
import torch
import torch.nn as nn
import random
import itertools
from agent import Agent

# BUFFER_SIZE = 500000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 100000
TARGET_UPDATE_FREQUENCY = 10

print("Gym version:", gym.__version__)
print("PyTorch version:", torch.__version__)
env = gym.make(id="CartPole-v1", render_mode="human")
# env = gym.make(id="CartPole-v1")
s, info = env.reset()

n_state = len(s)
n_action = env.action_space.n

"""Generate agents"""

agent = Agent(idx=0,
              n_input=n_state,
              n_output=n_action,
              mode='train')

"""Main Training Loop"""

n_episode = 300
n_time_step = 1000

REWARD_BUFFER = np.zeros(shape=n_episode)
for episode_i in range(n_episode):
    # for episode_i in itertools.count():
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY],
                            [EPSILON_START, EPSILON_END])  # interpolation
        random_sample = random.random()
        # print(random_sample)
        if random_sample <= epsilon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s)

        s_, r, done, timelimit, info = env.step(a)  # timelimit and info are not used in this case
        agent.memo.add_memo(s, a, r, done, s_)
        s = s_
        episode_reward += r

        if done:
            # print(step_i,done)
            s, info = env.reset()
            # print(f"{episode_i + 1} epi {step_i + 1} step:", done)
            REWARD_BUFFER[episode_i] = episode_reward
            break

        # render it once the average reward is greater than 50
        if np.mean(REWARD_BUFFER[:episode_i]) >= 60:
            count = 0
            print("Render starts.")
            while True:
                a = agent.online_net.act(s)
                s, r, done, timelimit, info = env.step(a)
                # print(count,a)
                count += 1
                env.render() # render the environment

                if done:
                    count = 0
                    env.reset() # reset the environment

        # Start Gradient Step
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()  # update batch-size amounts of Q

        # Compute Targets
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # ?
        targets = batch_r + agent.GAMMA * (1 - batch_done) * max_target_q_values

        # Compute Q_values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)  # ?

        # Compute Loss
        loss = nn.functional.smooth_l1_loss(a_q_values, targets)

        # Gradient Descent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    # Update target network
    # if episode_i % 1 == 0:
    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Print the training progress
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))

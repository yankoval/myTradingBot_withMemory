import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state,
    calcRewardExp,
    calcRewardLine
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10
                ,rewardFunc=None,trStrat = 'long' ):
    if callable(rewardFunc):
        rewardFunc = rewardFunc
    elif rewardFunc == 'calcRewardLine':
        rewardFunc = calcRewardLine
    else:
        rewardFunc = calcRewardExp
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1,agent.inventory )

    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):        
        reward = .5
        next_state = get_state(data, t + 1, window_size + 1, agent.inventory )

        # select an action
        action = agent.act(state)

        # ENTER(BUY or SELL depending of trStrat)
        if action == 1 and len(agent.inventory) == 0:
            agent.inventory.append(data[t])

        # CLOSE(BUY or SELL depending of trStrat)
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = (data[t] - bought_price if trStrat == 'long' 
                               else bought_price - data[t])
            reward = rewardFunc(bought_price if trStrat == 'long' 
                                else -1*bought_price,data[t])
            total_profit += delta

        # HOLD
        elif action == 0 and len(agent.inventory) > 0:
            bought_price = agent.inventory[-1]
            delta = (data[t] - bought_price if trStrat == 'long' 
                               else bought_price - data[t])
            reward = rewardFunc(bought_price if trStrat == 'long' 
                                else -1*bought_price,data[t])
            #total_profit += delta
        else:
            pass

        done = (t == data_length - 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 2 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug,
                   rewardFunc=None,trStrat = 'long'):
    if callable(rewardFunc):
        rewardFunc = rewardFunc
    elif rewardFunc == 'calcRewardLine':
        rewardFunc = calcRewardLine
    else:
        rewardFunc = calcRewardExp
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.inventory = []
    
    state = get_state(data, 0, window_size + 1, agent.inventory )

    for t in range(data_length):        
        reward = .5
        next_state = get_state(data, t + 1, window_size + 1, agent.inventory )
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1 and len(agent.inventory) == 0:
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at:{} , by {}".format(str(t),format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = rewardFunc(data[t], bought_price)#delta #max(delta, 0)
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at:{} ,by {} | Position: {}".format(str(t),
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        elif action == 0 and len(agent.inventory) > 0:
            bought_price = agent.inventory[0]
            delta = data[t] - bought_price
            reward = rewardFunc(data[t], bought_price)#delta #max(delta, 0)
            #total_profit += delta
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history

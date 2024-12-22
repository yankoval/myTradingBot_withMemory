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
    get_state2,
    get_state3,
    calcRewardExp,
    calcRewardLineShift05,
    calcRewardLineSigmoid
)
from scipy.special import expit
from trading_bot.utils import (Data1)

# TODO: Create universal broker object to do standard Buy Sell SL TP Profit calc
# TODO: Create and test model based on standard actions: "Enter with SL" and "Wait" to speed up training
# TODO: Create and test strategy besed on data optimased by mooving average: abs or % differens to SMA. Goal is to test is it more universal (test on different tikers)
# TODO: Create and test model based on additioanal laer of different indicators: Example SMA_233, SMA_14, SMA_7. Or SMA, RSI, OBV
# TODO: Create model with Vollume info (optionaly Fractal)


def train_model(agent, episode, data:Data1, ep_count=100, batch_size=32, window_size=10, rewardFunc = None, trStrat = 'long',
                get_state=get_state3, dataOHLCV_=None, brokerFee=None):
    if callable(rewardFunc):
        rewardFunc = rewardFunc
    elif rewardFunc == 'calcRewardLine':
        rewardFunc = calcRewardLineSigmoid
    else:
        rewardFunc = calcRewardExp
    total_profit = 0
    data_length = data.df.shape[0] - 1

    agent.inventory = []
    avg_loss = []
    startFrom = 800

    state = data.getState(window_size,agent.inventory,iloc=startFrom)
    #get_state(data, startFrom, window_size, memory=agent.inventory, dataOHLCV=dataOHLCV)

    # total=data_length,
    for t in tqdm(range(startFrom, data_length-2), leave=True, desc='Episode {}/{} epsilon:{}'.format(episode, ep_count,agent.epsilon)):
        reward = 0
        lastDealPrice = data.df.Close.iloc[t]
        stake = data.df.Close.iloc[startFrom]
        # next_state = get_state(data, t + 1, n_days=window_size, memory=agent.inventory, dataOHLCV=dataOHLCV)
# FIXME: Проверить логику next_state после внесения в стате данных о текущем профите
        # select an action
        action,actProbs = agent.act(state)
        if type(action) not in [np.int64, int]:
            raise
        # print(f'| t:{t}, action:{action}, epsilon: {agent.epsilon}',end="")

        # ENTER(BUY or SELL depending of trStrat)
        if action == 1 and len(agent.inventory) == 0:
            agent.inventory.append(lastDealPrice if trStrat.lower() in ['long', 'longShort'] else -lastDealPrice)
# FIXME: Правильно вычислять ревард для всех типов действий: Buy Sell Hold
            total_profit += -lastDealPrice*brokerFee if brokerFee else 0
            reward = rewardFunc(lastDealPrice, total_profit)
            # reward = max(total_profit / stake,0)

        # CLOSE(BUY or SELL depending of trStrat)
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = (lastDealPrice - bought_price if bought_price > 0
                               else bought_price + lastDealPrice)
            # if brokerFee:
            #     brokerFeeVal = lastDealPrice*brokerFee if trStrat not in ['longShort'] else lastDealPrice*brokerFee*2
            delta = delta - (lastDealPrice*brokerFee if brokerFee else 0)
            total_profit += delta
            reward = rewardFunc(lastDealPrice, total_profit)

            # This reward policy is cumulative
            # reward = max(delta, 0)
            # reward = max(total_profit / stake, 0)

        # HOLD
        elif action == 0 and len(agent.inventory) > 0:
            bought_price = agent.inventory[0]
            delta = lastDealPrice - (bought_price if bought_price > 0
                               else bought_price + lastDealPrice)
            # reward = max(delta, 0)
            reward = rewardFunc(lastDealPrice, total_profit+delta)
            # reward = max((total_profit+delta) / stake, 0)
            # total_profit += delta
        else:
            pass
        done = (t == data_length - 3)
        next_state = data.getState(window_size, agent.inventory, iloc=t + 1)
        #next_state = get_state(data, t + 1, n_days=window_size, memory=agent.inventory, dataOHLCV=dataOHLCV)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            # print(f'memory FULL on:{t}')
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 1 == 0:
        agent.save(episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug, startFrom:int=1,):
    bro = data.broker
    data.df[list(range(agent.action_size))]=np.nan
    data.df['cash'] = np.nan
    data.df['value'] = np.nan
    # get_state=get_state3, dataOHLCV=None, brokerFee=None,startFrom:int=0):
    # if callable(rewardFunc):
    #     rewardFunc = rewardFunc
    # elif rewardFunc == 'calcRewardLine':
    #     rewardFunc = calcRewardLineSigmoid
    # else:
    #     rewardFunc = calcRewardExp
    total_profit = 0
    maxDrawdownAbs = 0
    cumulativeDrawdownAbs =0 # Profit/Loss Summ on each step represent probability of profit
    data_length = data.df.shape[0] - 1

    history = []
    agent.inventory = []
    # state = get_state(data, startFrom, window_size, agent.inventory, dataOHLCV=dataOHLCV)
    state = data.getState(window_size, agent.inventory, iloc=startFrom)
    # state = data.getState()
    brokerFee = data.broker.getcommissioninfo(data)
    brokerFee = brokerFee.p.commission
    data.iloc = startFrom
    for t in tqdm(range(startFrom, data_length-2), leave=True, desc=f'Episode {startFrom}/{ data_length-2}.'):
        # tProgress = int(((t-startFrom) / ( data_length-2-startFrom)) * 100)
        # print('\r[' + '=' * tProgress + '_'*int(100-tProgress)+']', end='')
        lastDealPrice = data.df.Close.iloc[data.iloc]
        reward = .5
        pos = bro.getposition()
        data.next()
        try:
            next_state = data.getState(window_size, agent.inventory, iloc=data.iloc + 1)
        except:
            raise
# FIXME: Провериь логику получения состояния evaluate_model, в нем не учитывается действия текущего шага!
        # select an action
        action,action_probs = agent.act(state, is_eval=True)
        for i in range(agent.action_size):
            data.df.at[data.loc,i] = action_probs[0][i]
        #print(f'Action:{action}')
        # BUY
        if action == 1:# and len(agent.inventory) == 0:
            res = data.broker.buy()
            if res:
            # if brokerFee:
            #     # reward = rewardFunc(lastDealPrice, lastDealPrice*(1-brokerFee))
            #     # FIXME: Правильно вычислять ревард для всех типов действий: Buy Sell Hold
                total_profit += -lastDealPrice * brokerFee if brokerFee else 0
                total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
                agent.inventory.append(lastDealPrice)
                maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
                history.append((t, lastDealPrice, "BUY", total_profit, maxDrawdownAbs,data.df.iloc[t].name))
                if debug:
                    logging.debug("Enter at:{} , by {}".format(str(t),format_currency(lastDealPrice)))
        
        # SELL
        elif action == 2 and pos.size>0:
            res = data.broker.sell(size=pos.size)
            if res:
                # bought_price = agent.inventory.pop(0)
                bought_price = pos.price
            # if brokerFee:
            #     # reward = rewardFunc(lastDealPrice, lastDealPrice*(1-brokerFee))
            #     # FIXME: Правильно вычислять ревард для всех типов действий: Buy Sell Hold
                total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
                # reward = rewardFunc(pos.size*pos.price, data[0] + total_profit)
                maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
                history.append((t, lastDealPrice, "SELL", total_profit, maxDrawdownAbs,data.df.iloc[t].name))
                if debug:
                    logging.debug("Exit at:{} ,by {} | Position: {}".format(str(t),
                        format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        elif action == 0 and len(agent.inventory) > 0:
            delta = 0
            # if len(agent.inventory) > 0:
            # bought_price = agent.inventory[0]
            # delta = (lastDealPrice - bought_price if trStrat == 'long'
            #          else bought_price - lastDealPrice)
            # delta = delta - lastDealPrice * brokerFee if brokerFee else 0
            # total_profit += delta
            total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
            # maxDrawdownAbs = maxDrawdownAbs if total_profit+delta > maxDrawdownAbs else total_profit+delta
            maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
            history.append((t, lastDealPrice, "HOLD", total_profit, maxDrawdownAbs, data.df.iloc[t].name))

        data.df.at[data.loc, 'cash'] = bro.get_cash()
        data.df.at[data.loc, 'value'] = bro.getvalue()
        done = (t == data_length - 3)
        # agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        #print(f'total_profit:{total_profit}')
        if done:
            if  len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = lastDealPrice - bought_price
                delta = delta - lastDealPrice * brokerFee if brokerFee else 0
                total_profit += delta
            return total_profit, history, maxDrawdownAbs

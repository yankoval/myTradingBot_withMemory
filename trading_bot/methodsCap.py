# defaults

DEFAULT_SIZE = 1

import numpy as np
from tqdm import tqdm
from .ops import (
    get_state,
    get_state2,
    get_state3,
    calcRewardExp,
    calcRewardLineShift05,
    calcRewardLineSigmoid
)

from pathlib import Path
import logging

logger = logging.getLogger(__name__)
pass

# TODO: Create universal broker object to do standard Buy Sell SL TP Profit calc
# TODO: Create and test model based on standard actions: "Enter with SL" and "Wait" to speed up training
# TODO: Create and test strategy besed on data optimased by mooving average: abs or % differens to SMA. Goal is to test
#  is it more universal (test on different tikers)
# TODO: Create and test model based on additioanal laer of different indicators: Example SMA_233, SMA_14, SMA_7. Or SMA,
#  RSI, OBV
# TODO: Create model with Volume info (optionally Fractal)


def train_model(agent, data, *args, episode=None, ep_count=100, batch_size=32, window_size=10, reward_func=None,
                tr_strat='Long',
                get_state=get_state3, data_ohlcv_=None, broker_fee=None, start_from:int=0, **kwargs):
    logger = logging.getLogger(__name__)
    logger.debug(f'Starting train_model: {agent.model_name}, start with episode:{episode}.')
    agent.episode = episode if episode else agent.episode
    size = kwargs.get('size', DEFAULT_SIZE)
    if callable(reward_func):
        reward_func = reward_func
    elif reward_func == 'calcRewardLine':
        reward_func = calcRewardLineSigmoid
    else:
        reward_func = calcRewardExp
    total_profit = 0
    data_length = data.df.shape[0] - 1

    agent.inventory = []
    avg_loss = []
    # start_from = start_from #if type(start_from) is not str else data.df.index.get_loc(start_from).start

    state = data.getState(window_size, agent, iloc=start_from)
    #get_state(data, start_from, window_size, memory=agent.inventory, dataOHLCV=dataOHLCV)

    # total=data_length,
    for t in tqdm(range(start_from, data_length - 2), leave=True,
                  desc='Episode {}/{} epsilon:{}'.format(episode, ep_count, agent.epsilon)):
        # logger.debug(f'Starting episode:{episode}, model: {agent.model_name}.')
        reward = 0
        lastDealPrice = data.df.Close.iloc[t]
        stake = data.df.Close.iloc[start_from]
        # next_state = get_state(data, t + 1, n_days=window_size, memory=agent.inventory, dataOHLCV=dataOHLCV)
        # FIXME: Проверить логику next_state после внесения в стате данных о текущем профите
        # select an action
        action, actProbs = agent.act(state)
        assert type(action) in [np.int64, int], 'Action must be an integer'
        # BUY
        if action == 1:  # and len(agent.inventory) == 0:
            agent.inventory.append(-1 * size * lastDealPrice)
            # FIXME: Правильно вычислять ревард для всех типов действий: Buy Sell Hold
            total_profit, profit, pos, reward = data.getProfit(agent, t)

        # SELL
        elif action == 2:  # and len(agent.inventory) > 0:
            agent.inventory.append(size * lastDealPrice)
            total_profit, profit, pos, reward = data.getProfit(agent, t)

        # HOLD
        elif action == 0:  # and len(agent.inventory) > 0:
            total_profit, profit, pos, reward = data.getProfit(agent, t)
        else:
            raise ValueError(f'Action not in range:{action}')
        done = (t == data_length - 3)
        next_state = data.getState(window_size, agent, iloc=t + 1)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) > batch_size:
            # print(f'memory FULL on:{t}')
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state
    agent.episode += 1
    if episode % 1 == 0:
        agent.save(episode)
        logger.info(f'Episode {episode} saved, model:{agent.model_name}.')
    logger.info(f'Total profit: {total_profit}, episode: {episode}')
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug, *args, start_from=1, **kwargs):
    assert logger is not None, 'Evaluate logger not set'
    # start_from = start_from if type(start_from) is int else data.df.index.get_loc(start_from).start
    size = kwargs.get('size', DEFAULT_SIZE)
    bro = data.broker
    data.df[list(range(agent.action_size))] = np.nan

    data.df['BUY'] = np.nan
    data.df['SELL'] = np.nan

    data.df['cash'] = np.nan
    data.df['value'] = np.nan
    data.df['posAvgPrice'] = np.nan
    # total_profit_, profit, pos, reward
    data.df['total_profit_'] = np.nan
    data.df['profit_'] = np.nan
    data.df['pos_'] = np.nan
    data.df['reward_'] = np.nan
    data.df['pos'] = np.nan
    data.df['maxDrawdown'] = np.nan
    data.df['total_profit'] = np.nan

    total_profit, total_profit_, profit, pos = 0, 0, 0, 0
    maxDrawdownAbs = 0
    cumulativeDrawdownAbs = 0  # Profit/Loss Summ on each step represent probability of profit
    data_length = data.df.shape[0] - 1

    history = []
    agent.inventory = []
    # state = get_state(data, startFrom, window_size, agent.inventory, dataOHLCV=dataOHLCV)
    state = data.getState(window_size, agent, iloc=start_from)
    # state = data.getState()
    brokerFee = data.broker.getcommissioninfo(data)
    brokerFee = brokerFee.p.commission
    data.next(iloc=start_from)
    logger.info(f'Starting evaluation from {start_from}/{data.loc}')
    for t in tqdm(range(start_from + 1, data_length - 2), leave=True,
                  desc=f'Evaluate model, episode {start_from}/{data_length - 2}.'):
        data.next()
        currentDealPrice = data.df.Close.iloc[data.iloc]
        reward = .5
        # pos = bro.getposition()
        try:
            next_state = data.getState(window_size, agent, iloc=data.iloc + 1)
        except:
            raise
        # FIXME: Провериь логику получения состояния evaluate_model, в нем не учитывается действия текущего шага!
        # select an action
        action, action_probs = agent.act(state, is_eval=True)
        for i in range(agent.action_size):
            data.df.at[data.loc, i] = action_probs[0][i]
        #print(f'Action:{action}')
        # BUY
        if action == 1:  # and len(agent.inventory) == 0:
            res = data.broker.buy(size=size, exectype='takeOrCancel')
            if res:
                agent.inventory.append(-1 * size * currentDealPrice)
                data.df.at[data.loc, 'BUY'] = size
                # FIXME: Правильно вычислять ревард для всех типов действий: Buy Sell Hold
                total_profit_, profit, pos, reward = data.getProfit(agent, t)
                # total_profit += -currentDealPrice * brokerFee if brokerFee else 0
                total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
                # agent.inventory.append(currentDealPrice)
                maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
                history.append((t, currentDealPrice, "BUY", total_profit, maxDrawdownAbs, data.df.iloc[t].name))
                if debug:
                    logger.debug(
                        f'{data.loc}, bro.get_cash():{bro.get_cash()}, bro.getvalue(data):{bro.getvalue(data)}, '
                        f'bro.getposition():{bro.getposition()}')
                    logger.debug(f'total_profit_:{total_profit_}, profit:{profit}, pos:{pos}, reward:{reward},')
        # SELL
        elif action == 2:  # and pos.size>0:
            res = data.broker.sell(size=size, exectype='takeOrCancel')
            if res:
                # bought_price = pos.price
                agent.inventory.append(size * currentDealPrice)
                data.df.at[data.loc, 'SELL'] = size
                total_profit_, profit, pos, reward = data.getProfit(agent, t)
                total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
                # reward = rewardFunc(pos.size*pos.price, data[0] + total_profit)
                maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
                history.append((t, currentDealPrice, "SELL", total_profit, maxDrawdownAbs, data.df.iloc[t].name))
                if debug:
                    logger.debug(
                        f'{data.loc}, bro.get_cash():{bro.get_cash()}, bro.getvalue(data):{bro.getvalue(data)}, '
                        f'bro.getposition():{bro.getposition()}')
                    logger.debug(f'total_profit_:{total_profit_}, profit:{profit}, pos:{pos}, reward:{reward},'
                                 f'maxDrawdown:{maxDrawdownAbs}')
        # HOLD
        elif action == 0 and len(agent.inventory) > 0:
            delta = 0
            total_profit_, profit, pos, reward = data.getProfit(agent, t)
            if debug:
                logger.info(f'total_profit_:{total_profit_}, profit:{profit}, pos:{pos}, reward:{reward},')
            total_profit = bro.get_cash() + bro.getvalue(data) - bro.startingcash
            # maxDrawdownAbs = maxDrawdownAbs if total_profit+delta > maxDrawdownAbs else total_profit+delta
            maxDrawdownAbs = maxDrawdownAbs if total_profit > maxDrawdownAbs else total_profit
            history.append((t, currentDealPrice, "HOLD", total_profit, maxDrawdownAbs, data.df.iloc[t].name))
            if debug:
                logger.debug(
                    f'{data.loc}, bro.get_cash():{bro.get_cash()}, bro.getvalue(data):{bro.getvalue(data)}, '
                    f'bro.getposition():{bro.getposition()}')
                logger.debug(f'total_profit_:{total_profit_}, profit:{profit}, pos:{pos}, reward:{reward},')

        # Write history broker
        data.df.at[data.loc, 'cash'] = bro.get_cash()
        data.df.at[data.loc, 'value'] = bro.getvalue()
        data.df.at[data.loc, 'pos'] = bro.getposition().size
        data.df.at[data.loc, 'posAvgPrice'] = bro.getposition().price
        data.df.at[data.loc, 'total_profit'] = total_profit
        data.df.at[data.loc, 'maxDrawdown'] = maxDrawdownAbs
        # debug simple broker data
        data.df.at[data.loc, 'total_profit_'] = total_profit_
        data.df.at[data.loc, 'profit_'] = profit
        data.df.at[data.loc, 'pos_'] = pos
        data.df.at[data.loc, 'reward_'] = reward

        if round(abs(total_profit - (total_profit_ + profit)), 2) > 0:
            pass
            # logger.error('broker profit value error')
        if pos != bro.getposition().size:
            logger.error('broker profit qty error')
        if t // 10 == 0:
            logger.debug(f'Current state for step {t} is {data.broker.getStake(data)}')
        done = (t == data_length - 3)
        # agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        #print(f'total_profit:{total_profit}')
        if done:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = currentDealPrice - bought_price
                delta = delta - currentDealPrice * brokerFee if brokerFee else 0
                total_profit += delta

            logger.info(f'Evaluate results. periods:{t}/{data.loc}, getStake: {data.broker.getStake(data)}')
            return total_profit, history, maxDrawdownAbs
    raise AssertionError('Exit without done condition')
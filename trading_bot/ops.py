import os
import math
import logging

import numpy as np
import pandas as pd
from scipy.special import expit


def calcRewardExp(dealPrice=None, curPrice=None):
    d = (curPrice - dealPrice) / dealPrice if dealPrice > 0 else (-1 * dealPrice - curPrice) / (-1 * dealPrice)
    return (1 + .25 / (-d - .5)) * 1000 if d >= 0 else (.088 / (.5 - 2.718281 ** (d * 10 - 1)) - 0.176) * 1000


#  TODO: Проверить нормализацию calcRewardLine()
def calcRewardLineShift05(dealPrice, curPrice):
    return 0.5 + 100 * (
        (curPrice - dealPrice) / dealPrice if dealPrice >= 0 else (-1 * dealPrice - curPrice) / (-1 * dealPrice))


def calcRewardLineSigmoid(dealPrice, curPrice):
    return expit(
        (curPrice - dealPrice) / dealPrice if dealPrice >= 0 else (-1 * dealPrice - curPrice) / (-1 * dealPrice))


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data: pd.DataFrame, t: int, n_days=0, memory=None, dataOHLCV=None):
    """Returns an n-day state representation ending at time t
    """
    n_days = n_days - 1
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    res.append(sigmoid(data[t] - memory[-1]) if memory else 0.5)
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    # if t % 10 == 0:
    # print(np.array([res]),res,n_days,memory)
    return np.array([res])

    # Возвращает состояние фрактал нужного уровня + текущее значение


def get_state2(data: list, t: int, n_days=10, memory=None, dataOHLCV=None):
    """Returns an n-day state representation ending at time t
        """
    data = OHLCVtoSeries(fractalsExp(dataOHLCV.iloc[:t + 1], 3))
    n_days = n_days - 1
    d = t - n_days + 1
    block = data.iloc[-n_days:] if d >= 0 else -d * [data.iloc[0]] + data.iloc[0: t + 1]  # pad with t0
    res = []
    # Запишем текущий профит если сделка была
    res.append(sigmoid(dataOHLCV['Close'].iloc[t] - memory[-1]) if memory else 0.5)
    # Запишем данные из окна (после обработок)
    for i in range(n_days - 2):
        res.append(sigmoid(block[i + 1] - block[i]))
    # Запишем последнюю цену без обработок
    res.append(sigmoid(dataOHLCV['Close'].iloc[t] - dataOHLCV['Close'].iloc[t - 1]))
    return np.array([res])


# Возвращает состояние фрактал нужного уровня + текущее значение тест
def get_state3(data: list, t: int, n_days, memory, dataOHLCV):
    """Returns an n-day state representation ending at time t
    """
    data = OHLCVtoSeries(fractalsExp(dataOHLCV.iloc[:t], 3)) 
    # n_days = n_days - 1
    # d = t - n_days + 1
    block = data.iloc[-n_days:]  # if d >= 0 else -d * [data.iloc[0]] + data.iloc[0: t + 1]  # pad with t0
    res = []
    # Запишем текущий профит если сделка была
    res.append(expit(dataOHLCV['Close'].iloc[t] - memory[-1] if memory else 0))
    # Запишем цену последней сделки относительно цены последнего фрактала
    res.append(expit(dataOHLCV['Close'].iloc[t] - data.iloc[-1]))
    # Запишем данные цен фракталов относительно цен соответсвующих им предидущих фркталов
    res = res + data.iloc[-(n_days - len(res) + 1):].diff().iloc[-(n_days - len(res)):].apply(expit).to_list()
    return np.array([res])


def fractals(d: pd.DataFrame):
    d = d.assign(
        fractals_high=np.where((d['High'] >= d['High'].shift(1)) & (d['High'] >= d['High'].shift(2)) & (
                d['High'] > d['High'].shift(-1)) & (d['High'] > d['High'].shift(-2)),
                               True, False
                               ))
    d = d.assign(
        fractals_low=np.where(
            (d['Low'] <= d['Low'].shift(1)) & (d['Low'] <= d['Low'].shift(2)) & (d['Low'] < d['Low'].shift(-1)) & (
                    d['Low'] < d['Low'].shift(-2)),
            True, False
        ))
    return d


def fractalsExp(df: pd.DataFrame, n: int):
    res = df
    for i in range(n):
        res = fractals(res)
        res = res[res.fractals_high | res.fractals_low]
    return res


def OHLCVtoSeries(df: pd.DataFrame):
    df = pd.concat([df.High[df.fractals_high], df.Low[df.fractals_low]], axis=1)
    df = df.High.fillna(df.Low)
    return df

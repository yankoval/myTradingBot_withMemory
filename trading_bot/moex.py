#
# module for load market data from moex.com
# 12/24
#

# moexalgo
# eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0
#

# api
# candles futures 'https://iss.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/NGZ4/candles.json?from=2024-11-21&till=2024-11-21&interval=1'
# interval Период свечей:
#
# 1 - 1 мин
# 10 - 10 мин
# 60 - 1 час
# 24 - 1 день
# 7 - 1 неделя
# 31 - 1 месяц
validCandlesInterval =  ['1', '10', '60', '24', '7', '31']

# Examples:
# df = pd.read_csv(r'https://iss.moex.com/iss/engines/futures/markets/forts/securities/RIH5/trades.csv?previous_session=1', skiprows=2,header=0, skipfooter=12, delimiter=';')
# previous_session=1 что бы вывелась предидущая сессия
# &TRADENO=1925038371093115277 начать со сделки с указанным номером
# limit = 5000

import requests
import pandas as pd
import json
from datetime import datetime, date, timedelta
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

apiKey = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3MzQ4NzY0MTMsImlhdCI6MTczMjI4NDQxMywiYXV0aF90aW1lIjoxNzMyMjg0MDE0LCJqdGkiOiIyNmUwZjczNS02ODA1LTQ2OWQtOGQzZi00N2VkMGRhZmZkNjIiLCJpc3MiOiJodHRwczovL3NzbzIubW9leC5jb20vYXV0aC9yZWFsbXMvY3JhbWwiLCJhdWQiOlsiYWNjb3VudCIsImlzcyJdLCJzdWIiOiJmOjBiYTZhOGYwLWMzOGEtNDlkNi1iYTBlLTg1NmYxZmU0YmY3ZTozNmE1ZTczZS05OTViLTRiNTUtOWVlMS0zMmE5NGM3NTljZDUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJpc3MiLCJzZXNzaW9uX3N0YXRlIjoiYjBjOTdjYWUtNjVhMi00ODUyLTg4OTEtZTVmNDg2ZDAyNGU4IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBvZmZsaW5lX2FjY2VzcyBpc3NfYWxnb3BhY2sgcHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwYzk3Y2FlLTY1YTItNDg1Mi04ODkxLWU1ZjQ4NmQwMjRlOCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwibmFtZSI6ItCY0LLQsNC9INCa0LjRgdC10LvRkdCyIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMzZhNWU3M2UtOTk1Yi00YjU1LTllZTEtMzJhOTRjNzU5Y2Q1IiwiZ2l2ZW5fbmFtZSI6ItCY0LLQsNC9IiwiZmFtaWx5X25hbWUiOiLQmtC40YHQtdC70ZHQsiJ9.iwQXQrn6tU1R5Ek1HS_ccNiKNuTPs8toPkjjvxDN2OTvIU2kzUamsRYfrXMwmyRmFSvpByWc0E_1-uhzRuak0RsXJ9xKogCQGKiNEbBf7ELDA45FcQM56hhWOrMIYNDC57Gbi-stTELzMGmBijg5tt2vMn4q9VKaLoYMJJ3yuzgTnWz1VqpKCSAGYokT1k-17eQYSFpWPdR61G09-Nezy7P01XPGDuket0-9o5eUar15x07bwWrulB8VXZrAcQ7Z75XM5qJyUSXgtdenCHgmXV3b5WzfAoNrL5AXvutzk5YtyJEFpyi3oOUIjTXTuZGtPfB-DrXmtlpo7Ydm1O3QQw'
baseUrl = 'https://iss.moex.com/iss'

moexTypes = dict()
marketSheme = dict()

def getTyped(d:dict,s:str=''):
    """ return namedtuple structure from d in iss format with s key present in it"""
    if s:
        return namedtuple(s, d[s]['columns'])(*d[s]['data'][0])
    return namedtuple('sec', [k for k in d.keys() if d.get(k, {}).get('data')])(
        *[getTyped(d, s=k) for k in d.keys() if d.get(k, {}).get('data')])
def moexMarketSheme():
    """ read base market data like boards engines etc"""
    # index.xml
    url = f'{baseUrl}/index.json'
    headers = {
      'Authorization': f'Bearer {apiKey}',
    }
    # print(url)
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        try:
          # print(response.json)
          res = json.loads(response.text)
          if not res.get('engines', False):
              print(f'Error loading market sheme')
              raise Exception
          for mType in res.keys():
            moexTypes.update({mType: namedtuple(mType, res[mType]['columns'])})
            tmp =[]
            for row in res[mType]['data']:
                tmp.append(moexTypes[mType](*row))
            marketSheme.update({mType:tmp})
          else:
              return None
        except json.decoder.JSONDecodeError:
            print(f'Error parsr json responce: {response.text[:50]}')
    else:
        print(f'Error request responce : {response.status_code}')
        raise Exception
def loadLastDeals(secId:str="SBER"):
    """ Load last session dials, return pandas dataframe """
    secInf, engine, market, board, secInfTraded  = secInfo(secId)
    if secInf == None:
        return pd.DataFrame()
    # boardGroupInfo = [row for row in marketSheme['boardgroups'] if row.name == secInf.group]
    # if len(boardGroupInfo) != 1:
    #     return pd.DataFrame()
    # boardGroupInfo = boardGroupInfo[0]
    #
    # engine, market, boards = boardGroupInfo.trade_engine_name, boardGroupInfo.market_name, secInf.primary_boardid
    df = pd.read_csv(
        f'{baseUrl}/engines/{engine}/markets/{market}/securities/{secId}/trades.csv?previous_session=1',
        skiprows=2, header=0, skipfooter=12, delimiter=';')
    return df
intervalDict = {'1':pd.Timedelta('1min'),
                '10':pd.Timedelta('10min'),
                '60':pd.Timedelta('1h'),
                '24':pd.Timedelta('1D'),
                '7':pd.Timedelta('1W'),
                '31':pd.Timedelta('31D'),}
def convertInterval(interval):
    if interval in intervalDict.keys():
        return interval
    try:
        return [k for k,v in intervalDict.items() if v == pd.Timedelta(interval)][0]
    except Exception as e:
        logger.error(f'Wrong time interval {interval}')
        raise e
def loadCandlesPage(sec:str= 'SBER', dateFrom:date=None, dateTo:date =None, interval='60', start='0'):
    # load data in iss.moex.com format, sec = 'SBER', interval in ['1','10','60','24','7','31']
    dateFrom = date.today() - timedelta(days=1) if dateFrom is None else dateFrom
    dateTo = dateFrom + timedelta(days=1) if dateTo is None else dateTo
    if type(sec) is str:
        sec, engines, markets, boards, secInfoTraded  = secInfo(sec)
        logging.debug(sec, engines, markets, boards, secInfoTraded)
        if not isinstance(sec,tuple):
            raise Exception('Error loading candles page')
    interval = convertInterval(interval)
    assert interval in ['1','10','60','24','7','31'] # Check interval
    # print(dateFrom,dateTo)
    # apiKey = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3MzQ4NzY0MTMsImlhdCI6MTczMjI4NDQxMywiYXV0aF90aW1lIjoxNzMyMjg0MDE0LCJqdGkiOiIyNmUwZjczNS02ODA1LTQ2OWQtOGQzZi00N2VkMGRhZmZkNjIiLCJpc3MiOiJodHRwczovL3NzbzIubW9leC5jb20vYXV0aC9yZWFsbXMvY3JhbWwiLCJhdWQiOlsiYWNjb3VudCIsImlzcyJdLCJzdWIiOiJmOjBiYTZhOGYwLWMzOGEtNDlkNi1iYTBlLTg1NmYxZmU0YmY3ZTozNmE1ZTczZS05OTViLTRiNTUtOWVlMS0zMmE5NGM3NTljZDUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJpc3MiLCJzZXNzaW9uX3N0YXRlIjoiYjBjOTdjYWUtNjVhMi00ODUyLTg4OTEtZTVmNDg2ZDAyNGU4IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBvZmZsaW5lX2FjY2VzcyBpc3NfYWxnb3BhY2sgcHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwYzk3Y2FlLTY1YTItNDg1Mi04ODkxLWU1ZjQ4NmQwMjRlOCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwibmFtZSI6ItCY0LLQsNC9INCa0LjRgdC10LvRkdCyIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMzZhNWU3M2UtOTk1Yi00YjU1LTllZTEtMzJhOTRjNzU5Y2Q1IiwiZ2l2ZW5fbmFtZSI6ItCY0LLQsNC9IiwiZmFtaWx5X25hbWUiOiLQmtC40YHQtdC70ZHQsiJ9.iwQXQrn6tU1R5Ek1HS_ccNiKNuTPs8toPkjjvxDN2OTvIU2kzUamsRYfrXMwmyRmFSvpByWc0E_1-uhzRuak0RsXJ9xKogCQGKiNEbBf7ELDA45FcQM56hhWOrMIYNDC57Gbi-stTELzMGmBijg5tt2vMn4q9VKaLoYMJJ3yuzgTnWz1VqpKCSAGYokT1k-17eQYSFpWPdR61G09-Nezy7P01XPGDuket0-9o5eUar15x07bwWrulB8VXZrAcQ7Z75XM5qJyUSXgtdenCHgmXV3b5WzfAoNrL5AXvutzk5YtyJEFpyi3oOUIjTXTuZGtPfB-DrXmtlpo7Ydm1O3QQw'

    # url = "https://apim.moex.com/iss/datashop/algopack/eq/obstats.json?date=2024-10-15"
    url = (f'{baseUrl}/engines/{engines}/markets/{markets}/boards/{boards}/securities/{sec.secid}/candles.json?from={dateFrom}'
           f'&till={dateTo}&interval={interval}&start={start}')

    headers = {
      'Authorization': f'Bearer {apiKey}',
    }
    # print(url)
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        try:
          # print(response.json)
          return json.loads(response.text)
        except json.decoder.JSONDecodeError:
            print(f'Error parsr json responce: {response.text[:50]}')
    else:
        print(f'Error request responce : {response.status_code}')
        raise Exception

def loadCandlesDfPage(*args, **kwargs):
    # convert candles data in iss.moex.com format
    # print(args,kwargs)
    data = loadCandlesPage(*args, **kwargs)
    # print(data)
    df = pd.DataFrame(data['candles']['data'])
    df.columns = data['candles']['columns']
    df['begin'] = pd.to_datetime(df['begin'])
    df['end'] = pd.to_datetime(df['end'])
    df.set_index('begin', inplace=True)
    return df
def toDfCandles(func):
    def wrapper(*args, **kwargs):
      """ formatDf=True (default)  to get df """
      if kwargs.pop('formatDf', True):
        data = func(*args, **kwargs)
        assert type(data) is dict, 'data is not dict'
        assert 'candles' in data.keys(), 'data has no key candles'
        assert 'data' in data['candles'].keys(), "data['candles'] has no key data"
        if len(data['candles']['data']) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data['candles']['data'])
        df.columns = data['candles']['columns']
        df['begin'] = pd.to_datetime(df['begin'])
        df['end'] = pd.to_datetime(df['end'])
        df.set_index('begin', inplace=True)
        return df
      else:  # оригинальная или декорированная
        return func(*args, **kwargs)
    return wrapper
candles = toDfCandles(loadCandlesPage) # multiformat variant (df|list)


def secInfo(secid):
    """ securities information and data from iss return tuple secInf, engine, market, board, secInfoTraded  """
    # load data in iss.moex.com format
    # baseUrl = 'https://iss.moex.com/iss'
    # engines, markets, boards, interval = 'futures', 'forts', 'rfud', 60
    # print(dateFrom,dateTo)
    # url = "https://iss.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/MXZ4.json"
    #https://iss.moex.com/iss/securities.json?q=MXZ4
    # url = f'{baseUrl}/engines/{engines}/markets/{markets}/boards/{boards}/securities/{secid}.json'
    url = f'{baseUrl}/securities.json?q={secid}'
    headers = {
      'Authorization': f'Bearer {apiKey}',
    }
    # print(url)
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        try:
          # print(response.json)
          res = json.loads(response.text)
          if not res['securities']['data']:
              print(f'Error loading {secid}')
              return None
          row = [row for row in res['securities']['data'] if secid in [row[1],row[2],row[5]] ]
          if any(row):
            # typeSecuritiesFound = namedtuple('sec', res['securities']['columns'])
            res['securities']['data'] = row
            secInf = getTyped(res,'securities')
            boardGroupInfo = [row for row in marketSheme['boardgroups'] if row.name == secInf.group]
            if len(boardGroupInfo) != 1:
                return None
            boardGroupInfo = boardGroupInfo[0]
            engine, market, board = boardGroupInfo.trade_engine_name, boardGroupInfo.market_name, secInf.primary_boardid
            # return secInf, engine, market, board
          else:
              return None
        except json.decoder.JSONDecodeError:
            print(f'Error parsr json responce: {response.text[:50]}')
    else:
        print(f'Error request responce : {response.status_code}')
        raise Exception
    if secInf.is_traded == 1:
        url = f'{baseUrl}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}.json'
        response = requests.request("GET", url, headers=headers)
        if response.status_code == 200:
            try:
                # print(response.json)
                res = response.json()
                if not res.get('securities',{}).get('data',{}):
                    print(f'Error loading {secid}')
                    raise 'Error loading secInfoTraded'
                secInfTraded = getTyped(res)
            except json.decoder.JSONDecodeError:
                print(f'Error parsr json responce: {response.text[:50]}')
                raise Exception
        else:
            print(f'Error request responce : {response.status_code}')
            raise Exception
    else:
        secInfTraded = None
    return secInf, engine, market, board, secInfTraded
def futSeries():
    """ futures series information from iss """
  # https://iss.moex.com/iss/statistics/engines/futures/markets/forts/series.json
    # load data in iss.moex.com format
    baseUrl = 'https://iss.moex.com/iss'
    # dateFrom = date.today() - timedelta(days=1) if dateFrom is None else dateFrom
    # dateTo = dateFrom + timedelta(days=1) if dateTo is None else dateTo
    engines, markets, boards, interval = 'futures', 'forts', 'rfud', 60
    # print(dateFrom,dateTo)
    apiKey = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3MzQ4NzY0MTMsImlhdCI6MTczMjI4NDQxMywiYXV0aF90aW1lIjoxNzMyMjg0MDE0LCJqdGkiOiIyNmUwZjczNS02ODA1LTQ2OWQtOGQzZi00N2VkMGRhZmZkNjIiLCJpc3MiOiJodHRwczovL3NzbzIubW9leC5jb20vYXV0aC9yZWFsbXMvY3JhbWwiLCJhdWQiOlsiYWNjb3VudCIsImlzcyJdLCJzdWIiOiJmOjBiYTZhOGYwLWMzOGEtNDlkNi1iYTBlLTg1NmYxZmU0YmY3ZTozNmE1ZTczZS05OTViLTRiNTUtOWVlMS0zMmE5NGM3NTljZDUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJpc3MiLCJzZXNzaW9uX3N0YXRlIjoiYjBjOTdjYWUtNjVhMi00ODUyLTg4OTEtZTVmNDg2ZDAyNGU4IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBvZmZsaW5lX2FjY2VzcyBpc3NfYWxnb3BhY2sgcHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwYzk3Y2FlLTY1YTItNDg1Mi04ODkxLWU1ZjQ4NmQwMjRlOCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwibmFtZSI6ItCY0LLQsNC9INCa0LjRgdC10LvRkdCyIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMzZhNWU3M2UtOTk1Yi00YjU1LTllZTEtMzJhOTRjNzU5Y2Q1IiwiZ2l2ZW5fbmFtZSI6ItCY0LLQsNC9IiwiZmFtaWx5X25hbWUiOiLQmtC40YHQtdC70ZHQsiJ9.iwQXQrn6tU1R5Ek1HS_ccNiKNuTPs8toPkjjvxDN2OTvIU2kzUamsRYfrXMwmyRmFSvpByWc0E_1-uhzRuak0RsXJ9xKogCQGKiNEbBf7ELDA45FcQM56hhWOrMIYNDC57Gbi-stTELzMGmBijg5tt2vMn4q9VKaLoYMJJ3yuzgTnWz1VqpKCSAGYokT1k-17eQYSFpWPdR61G09-Nezy7P01XPGDuket0-9o5eUar15x07bwWrulB8VXZrAcQ7Z75XM5qJyUSXgtdenCHgmXV3b5WzfAoNrL5AXvutzk5YtyJEFpyi3oOUIjTXTuZGtPfB-DrXmtlpo7Ydm1O3QQw'

    # url = "https://apim.moex.com/iss/datashop/algopack/eq/obstats.json?date=2024-10-15"
    url = f'{baseUrl}/statistics/engines/{engines}/markets/{markets}/series.json'

    headers = {
      'Authorization': f'Bearer {apiKey}',
    }
    # print(url)
    response = requests.request("GET", url, headers=headers)
    if response.status_code == 200:
        try:
          # print(response.json)
          return json.loads(response.text)
        except json.decoder.JSONDecodeError:
            print(f'Error parsr json responce: {response.text[:50]}')
    else:
        print(f'Error request responce : {response.status_code}')
        raise Exception
def toDfSeries(func):
    def wrapper(*args, **kwargs):
      if kwargs.pop('formatDf', True):
        data = func(*args, **kwargs)
        key = list(data.keys())[0] # get main key
        df = pd.DataFrame(data[key]['data'])
        df.columns = data[key]['columns']
        # df['begin'] = pd.to_datetime(df['begin'])
        # df['end'] = pd.to_datetime(df['end'])
        # df.set_index('begin', inplace=True)
        return df
      else:  # оригинальная или декорированная
        return func(*args, **kwargs)
    return wrapper

#def
series = toDfSeries(futSeries) # multiformat variant candles(sec=series(formatDf=False)['series']['data'][0][0]) futSeries.__doc__
moexMarketSheme()
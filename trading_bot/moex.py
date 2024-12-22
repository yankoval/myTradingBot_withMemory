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

import requests
import pandas as pd
import json
from datetime import datetime, date, timedelta

apiKey = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3MzQ4NzY0MTMsImlhdCI6MTczMjI4NDQxMywiYXV0aF90aW1lIjoxNzMyMjg0MDE0LCJqdGkiOiIyNmUwZjczNS02ODA1LTQ2OWQtOGQzZi00N2VkMGRhZmZkNjIiLCJpc3MiOiJodHRwczovL3NzbzIubW9leC5jb20vYXV0aC9yZWFsbXMvY3JhbWwiLCJhdWQiOlsiYWNjb3VudCIsImlzcyJdLCJzdWIiOiJmOjBiYTZhOGYwLWMzOGEtNDlkNi1iYTBlLTg1NmYxZmU0YmY3ZTozNmE1ZTczZS05OTViLTRiNTUtOWVlMS0zMmE5NGM3NTljZDUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJpc3MiLCJzZXNzaW9uX3N0YXRlIjoiYjBjOTdjYWUtNjVhMi00ODUyLTg4OTEtZTVmNDg2ZDAyNGU4IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBvZmZsaW5lX2FjY2VzcyBpc3NfYWxnb3BhY2sgcHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwYzk3Y2FlLTY1YTItNDg1Mi04ODkxLWU1ZjQ4NmQwMjRlOCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwibmFtZSI6ItCY0LLQsNC9INCa0LjRgdC10LvRkdCyIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMzZhNWU3M2UtOTk1Yi00YjU1LTllZTEtMzJhOTRjNzU5Y2Q1IiwiZ2l2ZW5fbmFtZSI6ItCY0LLQsNC9IiwiZmFtaWx5X25hbWUiOiLQmtC40YHQtdC70ZHQsiJ9.iwQXQrn6tU1R5Ek1HS_ccNiKNuTPs8toPkjjvxDN2OTvIU2kzUamsRYfrXMwmyRmFSvpByWc0E_1-uhzRuak0RsXJ9xKogCQGKiNEbBf7ELDA45FcQM56hhWOrMIYNDC57Gbi-stTELzMGmBijg5tt2vMn4q9VKaLoYMJJ3yuzgTnWz1VqpKCSAGYokT1k-17eQYSFpWPdR61G09-Nezy7P01XPGDuket0-9o5eUar15x07bwWrulB8VXZrAcQ7Z75XM5qJyUSXgtdenCHgmXV3b5WzfAoNrL5AXvutzk5YtyJEFpyi3oOUIjTXTuZGtPfB-DrXmtlpo7Ydm1O3QQw'
baseUrl = 'https://iss.moex.com/iss'

def loadCandlesPage(sec:str= 'SBER', dateFrom:date=None, dateTo:date =None, interval='60', start='0'):
    # load data in iss.moex.com format, sec = 'SBER', interval in ['1','10','60','24','7','31']
    dateFrom = date.today() - timedelta(days=1) if dateFrom is None else dateFrom
    dateTo = dateFrom + timedelta(days=1) if dateTo is None else dateTo
    engines, markets, boards = 'futures', 'forts', 'rfud'
    assert  interval in ['1','10','60','24','7','31'] # Check interval
    # print(dateFrom,dateTo)
    apiKey = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJaVHA2Tjg1ekE4YTBFVDZ5SFBTajJ2V0ZldzNOc2xiSVR2bnVaYWlSNS1NIn0.eyJleHAiOjE3MzQ4NzY0MTMsImlhdCI6MTczMjI4NDQxMywiYXV0aF90aW1lIjoxNzMyMjg0MDE0LCJqdGkiOiIyNmUwZjczNS02ODA1LTQ2OWQtOGQzZi00N2VkMGRhZmZkNjIiLCJpc3MiOiJodHRwczovL3NzbzIubW9leC5jb20vYXV0aC9yZWFsbXMvY3JhbWwiLCJhdWQiOlsiYWNjb3VudCIsImlzcyJdLCJzdWIiOiJmOjBiYTZhOGYwLWMzOGEtNDlkNi1iYTBlLTg1NmYxZmU0YmY3ZTozNmE1ZTczZS05OTViLTRiNTUtOWVlMS0zMmE5NGM3NTljZDUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJpc3MiLCJzZXNzaW9uX3N0YXRlIjoiYjBjOTdjYWUtNjVhMi00ODUyLTg4OTEtZTVmNDg2ZDAyNGU4IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIvKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6Im9wZW5pZCBvZmZsaW5lX2FjY2VzcyBpc3NfYWxnb3BhY2sgcHJvZmlsZSBlbWFpbCIsInNpZCI6ImIwYzk3Y2FlLTY1YTItNDg1Mi04ODkxLWU1ZjQ4NmQwMjRlOCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwiaXNzX3Blcm1pc3Npb25zIjoiMTM3LCAxMzgsIDEzOSwgMTQwLCAxNjUsIDE2NiwgMTY3LCAxNjgsIDMyOSwgNDIxIiwibmFtZSI6ItCY0LLQsNC9INCa0LjRgdC10LvRkdCyIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiMzZhNWU3M2UtOTk1Yi00YjU1LTllZTEtMzJhOTRjNzU5Y2Q1IiwiZ2l2ZW5fbmFtZSI6ItCY0LLQsNC9IiwiZmFtaWx5X25hbWUiOiLQmtC40YHQtdC70ZHQsiJ9.iwQXQrn6tU1R5Ek1HS_ccNiKNuTPs8toPkjjvxDN2OTvIU2kzUamsRYfrXMwmyRmFSvpByWc0E_1-uhzRuak0RsXJ9xKogCQGKiNEbBf7ELDA45FcQM56hhWOrMIYNDC57Gbi-stTELzMGmBijg5tt2vMn4q9VKaLoYMJJ3yuzgTnWz1VqpKCSAGYokT1k-17eQYSFpWPdR61G09-Nezy7P01XPGDuket0-9o5eUar15x07bwWrulB8VXZrAcQ7Z75XM5qJyUSXgtdenCHgmXV3b5WzfAoNrL5AXvutzk5YtyJEFpyi3oOUIjTXTuZGtPfB-DrXmtlpo7Ydm1O3QQw'

    # url = "https://apim.moex.com/iss/datashop/algopack/eq/obstats.json?date=2024-10-15"
    url = (f'{baseUrl}/engines/futures/markets/forts/boards/rfud/securities/{sec}/candles.json?from={dateFrom}'
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
    """ securities information and data from iss """
    # load data in iss.moex.com format
    baseUrl = 'https://iss.moex.com/iss'
    engines, markets, boards, interval = 'futures', 'forts', 'rfud', 60
    # print(dateFrom,dateTo)
    # url = "https://iss.moex.com/iss/engines/futures/markets/forts/boards/rfud/securities/MXZ4.json"
    url = f'{baseUrl}/engines/{engines}/markets/{markets}/boards/{boards}/securities/{secid}.json'

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
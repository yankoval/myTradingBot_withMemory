import os
import math
import logging
from pathlib import Path
import pandas as pd
import numpy as np
# import keras.backend as K
import matplotlib.pyplot as plt
from scipy.special import expit
from qbroker.broker import Position
from trading_bot.ops import OHLCVtoSeries,fractalsExp,get_state3
from trading_bot.moex import candles
from dateutil import parser
from sklearn.cluster import KMeans

# Formats Position
format_position = lambda price: ('$' if price < 0 else '$') + '{0:.2f}'.format(price)

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))

logger = logging.getLogger('train')

plt.set_loglevel('DEBUG')

def calculateFractalsPairs(df, CalculateValues=True, CalculateTimeDiff=True):
    """Calculate fractals with granted pairs sequence"""
    # df['fHigh'] = np.where(
    #   (df['High'] > df['High'].shift(1)) &
    #   (df['High'] >= df['High'].shift(-1)) &
    #   (df['High'] > df['High'].shift(2)) &
    #   (df['High'] >= df['High'].shift(-2))
    #   , np.True_, np.False_
    #   )

    df['fLow'] = np.where(
        (df['Low'] < df['Low'].shift(1)) &
        (df['Low'] <= df['Low'].shift(-1)) &
        (df['Low'] < df['Low'].shift(2)) &
        (df['Low'] <= df['Low'].shift(-2))
        , np.True_, np.False_
    )
    # Calculate High fractal coresponded all Low froctals
    df['fHigh'] = np.where(df.fLow, df.index.to_series().values.astype("float64"),np.nan)
    df['fHigh'] = df['fHigh'].fillna(method='ffill')
    df['fHigh'] = np.where(df.index == df.groupby(['fHigh'])['High'].transform('idxmax'), True, False)
def getStateFractalsValues(df):
    """ Calculate fractal values and timedeltas """
    frac =  pd.DataFrame({'Close':pd.concat([df[df.fLow].Low,
               pd.Series(np.array(df[df.fHigh].High),
                         index=np.where(df[df.fHigh].fLow & df[df.fHigh].fHigh,
                                    df[df.fHigh].index + np.timedelta64(1,'s'),
                                    df[df.fHigh].index))],
              verify_integrity=True, sort=True)}
            )
    frac.sort_index(inplace=True)
    # calculate time difference in minets. Add 0 to begin of array to fit to original df shape
    frac['tDiff'] = np.append([1], np.diff(frac.index.values.astype('datetime64[s]')).astype('int'))
    return frac
def calcLevels(df,kMeansKwargs={"init":"k-means++", "n_init": 4, "n_clusters": 20}, type='futures',MINSTEP=25,DECIMALS=2):
    """ db scan setting for level clusterization
     lernValidateRatio ratio of records out of the forecastiong for validation"""
    epsLev0, epsLev1  = 800, 0.03 # 0.039
    fh = df.loc[df.fHigh==True].High.values
    fl = df.loc[df.fLow==True].Low.values
    # levels = []

    # Keans
    kmeans = KMeans(**kMeansKwargs) # , n_clusters=20, n_init=4
    # x = fl.to_list()#fh.to_list()#+fl.to_list()
    a = kmeans.fit(np.reshape(fh,(len(fh),1)))
    fh = np.sort(np.transpose(a.cluster_centers_)[0]).tolist()
    # x = fh.to_list()#+fl.to_list()
    b = kmeans.fit(np.reshape(fl,(len(fl),1)))
    fl = np.sort(np.transpose(b.cluster_centers_)[0]).tolist()
    levels =  fl+fh

    try:  # ls
        if type in ['futures_forts', 'futures']:
            levels = list(map(lambda x: int((x // MINSTEP) * MINSTEP),
                              levels))  # round as ticker price step
        elif type in ['stock_index', 'stock_index_eq', 'stock_shares', 'common_share']:
            levels = list(map(lambda x: round(x, DECIMALS), levels))
    except:
        pass  # in INDEX tikers no tikInfo.MINSTEP
    # chose 10 levels nearest to ticker last close value
    # close = df.iloc[-1].Close
    # levels = sorted(levels, key=lambda x: abs(x - close) / close)[:10]
    levels = sorted(levels)  # final sort to prevent
    return levels
def calcLevelsForEachInterval(df:pd.DataFrame,freq:str='D'):
    """ calculate df with sets of levels for each time step by tDiff as index"""
    # Generate day range index
    idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, normalize=True)
    # generate levels for each
    levels = [calcLevels(df.loc[:day + pd.Timedelta('1'+freq)]) for day in idx]
    # correct time shift to prevent future vision
    levels = [levels[0]] + levels[:-1]
    return pd.DataFrame(levels,index=idx)
def pltHist(dfin, hist, fName=None,startFrom=0):
    shift = startFrom #hist[0][0] if hist else 0
    df = dfin #.copy()
    df = df.iloc[shift:]
    b = [[r[0] for r in hist if r[2] == 'BUY'], [r[1] for r in hist if r[2] == 'BUY']]
    print(b)
    s = [[r[0] for r in hist if r[2] == 'SELL'], [r[1] for r in hist if r[2] == 'SELL']]
    print(s)

    # Calc capital stake + profit on each step
    df = df.assign(curCash=np.nan)
    df = df.assign(curPosQty=np.nan)
    df = df.assign(portfolio=np.nan)
    df = df.assign(maxDD=np.nan)
    profit = 0
    curPosQty, dealQty, curPos, maxDD = 0, 10, 0, 0
    curCash = df.iloc[0].Close * dealQty
    privStep = 0
    for i,step in enumerate(hist):
        df.iloc[privStep:step[0] - shift, df.columns.get_loc("curCash")] = curCash
        df.iloc[privStep:step[0] - shift, df.columns.get_loc("curPosQty")] = curPosQty
        df.iloc[privStep:step[0] - shift, df.columns.get_loc("portfolio")] = curCash + curPos
        df.iloc[privStep:step[0] - shift, df.columns.get_loc("maxDD")] = step[4]
        if step[2] == 'BUY':
            curCash -= df.iloc[step[0]-shift].Close * dealQty
            curPos += df.iloc[step[0]-shift].Close * dealQty
            curPosQty += dealQty
        if step[2] == 'SELL':
            curCash += df.iloc[step[0]-shift].Close * curPosQty
            curPos = 0
            curPosQty = 0
        if step[2] == 'HOLD':
            curPos = df.iloc[step[0] - shift].Close * curPosQty
        privStep = step[0] - shift
    # Fill df after last history record
    for i in list(range(hist[-1][0]-shift,df.shape[0])):
        curPos = df.iloc[i].Close * curPosQty
        df.iloc[i, df.columns.get_loc("curCash")] = curCash
        df.iloc[i, df.columns.get_loc("curPosQty")] = curPosQty
        df.iloc[i, df.columns.get_loc("portfolio")] = curCash + curPos
        df.iloc[i, df.columns.get_loc("maxDD")] = hist[-1][4]

    df.index = df.index.map(str)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(shift, df.shape[0], df.shape[0] / 5)
    major_ticks = df.iloc[major_ticks].index
    minor_ticks = np.arange(shift, df.shape[0], df.shape[0] / 10)
    minor_ticks = df.iloc[minor_ticks].index

    dRange = df.High.max() - df.Low.min()
    major_ticks_y = np.arange(df.Low.min(), df.High.max(), dRange / 6)
    minor_ticks_y = np.arange(df.Low.min(), df.High.max(), dRange / 12)


    try:
        fig, (ax0, ax1, axmdd,axProb,axCash,axValue, ax2) = plt.subplots(7, 1, sharex=True, gridspec_kw={'height_ratios':[1,1,1,1,1,1,6]}
                                            , figsize=(df.shape[0]//15, 8))
        ax0.plot(df.index, df.portfolio)
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax0.grid(which='both')
        ax1.plot(df.index, df.curPosQty)
        # make these tick labels invisible
        plt.setp(ax1.get_xticklabels(), visible=False)
        axmdd.plot(df.index, df.maxDD)
        # make these tick labels invisible
        plt.setp(axmdd.get_xticklabels(), visible=False)
        for i in range(3):
            axProb.plot(df.index, df[i],label=f'{i}')
        axProb.legend()
        axCash.plot(df.index, df.cash)
        axValue.plot(df.index, df.value)
        ax2.plot(df.index, df.High, 'g')
        ax2.plot(df.index, df.Low, 'y')
        ax2.scatter(df.iloc[list(map(lambda x:x-shift, b[0]))].index, b[1], marker='^', s=200, color='green', alpha=0.5)
        ax2.scatter(df.iloc[list(map(lambda x:x-shift, s[0]))].index, s[1], marker='v', s=200, color='red', alpha=0.3)

        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.set_yticks(major_ticks_y)
        ax2.set_yticks(minor_ticks_y, minor=True)

        # And a corresponding grid
        ax2.grid(which='both')

        # Or if you want different settings for the grids:
        ax2.text(0.01,0.9,fName, transform=ax2.transAxes) # ,df.High.max()-dRange/10
        ax0.text(0.01, 0.9,
                 f'portfolio, max:{df.portfolio.max():0.2f}, '
                 f'start:{df.Close.iloc[startFrom]*dealQty:0.2f}'
                 f', end:{df.portfolio.iloc[-1]:0.2f}, '
                 f'%{(df.portfolio.iloc[-1]-df.portfolio.iloc[0])/(df.portfolio.iloc[0]):0.2f}'
                 , transform=ax0.transAxes)
        ax1.text(0.01, 0.9, 'curPosQty', transform=ax1.transAxes)
        maxDD = df.maxDD.min()
        axmdd.text(0.01, 0.9, f'Max Drop Down Abs:{maxDD:0.2f}, :{maxDD/df.portfolio.iloc[0]:0.2f}%', transform=axmdd.transAxes)


        if fName:
            plt.savefig(Path.cwd() / 'graphs' / (fName + '.svg'), figsize=(39, 5))
    except Exception as e:
        logger.error(f'pltHist error:{str(e)}')

def show_train_result(result, val_position, initial_offset, history=None, df=pd.DataFrame(), modelName=None
                      , maxDrawdownAbs=None
                      ,startFrom:int=0):
    """ Displays training results
    """
    data = df
    df = data.df

    b = [[],[]]
    if history:
        # print(filter(lambda l: l[2]!='HOLD',history))
        pltHist(df, history, f'{modelName if modelName else "Graph"}_{result[0]:03d}', startFrom=startFrom)
        b = [[r[0] for r in history if r[2] == 'BUY'], [r[1] for r in history if r[2] == 'BUY']]
        s = [[r[0] for r in history if r[2] == 'SELL'], [r[1] for r in history if r[2] == 'SELL']]
    else:
        print('No deals!!!')
    lText = f'Episode {result[0]}/{result[1]} - Train Position: {format_position(result[2])} '
    lText = lText + f' Val Position: ' \
        f'{"USELESS" if val_position == initial_offset or val_position == 0.0 else format_position(val_position)}, '\
        f'maxDrawdownAbs:${maxDrawdownAbs:.2f}, '

    try:
        lText = lText + f' last BUY:{b[1][-1]} last close: {df.iloc[-1].Close}, BUY qty: {len(b[0])} ' \
                        f'SELL qty:{len(s[0])} positionL {len(b[0])-len(s[0])}'
        lText = lText + f' Train Loss: {result[3]:.4f} , buy deals qty: {len(b[0]) if b else 0}.'
    except:
        pass
    logger.info(lText)
    logger.debug(f'Buy history:{b}')


def show_eval_result(model_name, profit, initial_offset, hist=[]):
    """ Displays eval results
    """
    if profit == initial_offset or profit == 0.0:
        logger.info('{}: USELESS\n'.format(model_name))
    else:
        logger.info(f'{model_name}: {format_position(profit)}, len:{len(hist)}')


def get_stock_data(stock_file, tfCounts=None, tik=None, tFrame='daily', dFrom=None, dTo=None):
    """Reads stock data from csv file if stock_file specified else read from finam csv db
    """
    if not stock_file:
        return list(readFinam(tik, tFrame=tFrame, dFrom=dFrom, dTo=dTo)['Close'].iloc[:tfCounts])
    else:
        df = pd.read_csv(stock_file)
        return list(df.iloc[:tfCounts])

class Data:
    ver = 'v01'
    description = 'Base class'

    def __init__(self,*args,**kwargs):
        self.tik = self._name  = kwargs.get('tik')
        if args[0] and Path(args[0]).is_file():
            self.df = readData(args[0],None,*args[1:], **kwargs)
        elif Path(args[0]).is_dir():
            self.df = readData(None,args[0],*args[1:],**kwargs)
        else:
            raise ('Error: vrong arguments, Data("Path to file") OR Data("Path to DB", tik="SBER")')
        self.iloc = 0
        self.loc = self.df.index[self.iloc]
        self._prepareDf()
        self.ver = 'vd0'
    def _prepareDf(self):
        pass
    def getState(self,loc=None,*args,iloc=None,**kwargs):
        if iloc:
            return self.df.iloc[iloc]
        else:
            return self.df.loc[loc if loc else self.loc]

    def __repr__(self):
        return f'{self.ver}_{self.tik if self.tik else "NoTikerGiven"}'
    def __str__(self):
        return self.__repr__()
    def getDf(self,loc=None,iloc=None):
        if loc: return self.df.loc[loc:]
        if iloc: return self.df.iloc[iloc:]
        return self.df.loc[self.loc:]

    def next(self,loc=None,iloc=None):
        if loc:
            self.loc = loc
            iloc = self.df.index.get_loc(self.loc)
            assert iloc >= self.iloc
            self.iloc = iloc
            return
        if iloc:
            assert iloc >= self.iloc
            self.iloc = iloc
            self.loc = self.df.index[self.iloc]
            return
        if self.iloc + 1 <= self.df.shape[0]:
            self.iloc += 1
            self.loc = self.df.index[self.iloc]
    def setBroker(self,bro):
        self.broker = bro
        if not bro.data:
            bro.data = self
        bro.datas.append(self)
        bro.positions.update({self._name:Position()})
    def setsizer(self, sizer):
        '''
        Replace the default (fixed stake) sizer
        '''
        self._sizer = sizer
        sizer.set(self, self.broker)
        return sizer






# read data in panda DF OHLCV format
def readData(stock_file:str,dataPath:str,*agrs,tfCounts=None,tik=None,tFrame='daily',dFrom=None,dTo=None,skiprows=range(1, 1),nrows=None,**kwargs):
    """Reads stock data from csv file if stock_file specified else read from finam csv db
    """
    logger.info(f'stock_file: {stock_file}')
    if not stock_file:
        df = readFinam(tik
                         ,dataPath
                         ,*agrs
                         ,tFrame=tFrame
                         ,dFrom=dFrom
                         ,dTo=dTo
                         ,skiprows=skiprows
                         ,nrows=nrows
                         ,**kwargs
                         )
    else:
        df = pd.read_csv(stock_file
                         ,*agrs
                         ,index_col=['Date']
                         ,parse_dates=['Date']
                         ,skiprows=skiprows
                         ,nrows=nrows
                         ,**kwargs
                         ).loc[dFrom:dTo]
    df = df.iloc[:tfCounts]
    return df


def readFinam(tik,dataPath:str,*agrs,tFrame='daily',dFrom=None,dTo=None,skiprows=range(1, 1),nrows=None,**kwargs):
    """ read from local db or from moex"""
    tFrame={'min':'1','daily':'24','hourly':'60','minute':'1','monthly':31,'weekly':'7'}[tFrame]
    df = pd.DataFrame()
    # Load data page by pege till got empty data
    for timeout in range(100):
        logger.debug(f'df.shape{df.shape}, {df.index.max()}')
        dfTmp = candles(sec=tik, interval=tFrame,
                        dateFrom=parser.parse(dFrom), #datetime.now() - timedelta(days=days),
                        dateTo=parser.parse(dTo), #datetime.now() + timedelta(days=1),
                        start=str(df.shape[0])
                        )
        if dfTmp.shape[0] == 0:  # chek df is empty then exit
            logger.debug('Finished loading.')
        df = pd.concat([df, dfTmp])

    for i, name in enumerate(['Open', 'Close', 'High', 'Low', 'value', 'Volume']):
        df.rename(columns={df.columns[i]: name}, inplace=True)

    # Set the combined date-time as the index
    df.index.name = 'Date'

    # We need df with standart OHLCV columns only
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df
    # dataPath = Path(dataPath) #if dataPath else Path(r'D:/share/finam/data/')
    # return pd.read_csv(Path(dataPath, tik, tFrame, (tik + '.csv'))
    #                    , *agrs
    #                    , index_col=['Date']
    #                    , parse_dates=['Date']
    #                    , skiprows=skiprows
    #                    , nrows=nrows
    #                    , **kwargs
    #                    ).loc[dFrom:dTo]


def prepareData(df):
    return list(df['Close'])


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == "tensorflow":
        logger.debug("switching to TensorFlow for CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def fractalHigh(d):
    # for _ in range(3):
     return d.where(((d >= d.shift(1)) & (d  >= d.shift(2))
                    & (d > d.shift(-1)) & (d  > d.shift(-2)))
                 # ,True, False)
                 ).dropna()
    # return d
def fractalLow(d):
    return d.where((d < d.shift(1)) & (d < d.shift(2))
                   & (d <= d.shift(-1)) & (d <= d.shift(-2))
                # ,True, False
                ).dropna()

def prepAtr(df,period=13):
    df['TR'] = df['High'] - df['Low']
    df['ATR'] = df['TR'].rolling(window=period).mean()

def prepMa(df):
    df['MA'] = df['Close'].rolling(window = 233).mean()
    # df['CloseMAABS'] = df.Close - df.MA
    # df['CloseMAAT'] = df.CloseMAABS / df.MA
    # df['CloseMAATEXP'] = expit(df.CloseMAAT)

    return df

def prepFractals(df,windowSize=0):
    df['FH'] = [fractalHigh(fractalHigh(win)) for win in df.High.rolling(window=800)]
    df['FH'] = df.FH.map(lambda x: x[-windowSize:])
    for row in df.itertuples(): row.FH.index = row.FH.index +pd.Timedelta(microseconds=1)
    df['FHAT'] = ((df.FH - df.MA) / df.MA)
    df['FHATEXP'] = df.FHAT.map(expit)

    df['FL'] = [fractalLow(fractalLow(win)) for win in df.Low.rolling(window=800)]
    df['FL'] = df.FL.map(lambda x: x[-windowSize:])
    df['FLAT'] = ((df.FL - df.MA) / df.MA)
    df['FLATEXP'] = df.FLAT.map(expit)


    df['MIX'] =[ pd.concat([row.FHATEXP,row.FLATEXP]).sort_index() for row in df.itertuples()]
    df['MIX'] = df.MIX.map(lambda x: x[-windowSize:])
class Data1(Data):
    ver = 'v02'
    description = 'Fractals 2 level, MA, AT, EXP,State window_size=16'

    def _prepareDf(self):
        prepMa(self.df)
        prepAtr(self.df)
        # prepFractals(self.df)
        self.df['MIX'] = [OHLCVtoSeries(fractalsExp(win, 3))for win in self.df.rolling(window=800)]
        # self.prepData = prepareData(self.df)
    def getState(self,n_days,memory,loc=None,*args,iloc=None,**kwargs):
        # def get_state3(data: list, t: int, n_days=10, memory=None, dataOHLCV=None):
        # data = OHLCVtoSeries(fractalsExp(dataOHLCV.iloc[:t], 3))
        return get_state3(['self.prepData'],iloc,n_days=n_days,memory=memory,dataOHLCV=self.df)
        data = super().getState(loc, *args, iloc, **kwargs).MIX
            # n_days = n_days - 1
            # d = t - n_days + 1
        block = list(data.iloc[-n_days+2:])  # if d >= 0 else -d * [data.iloc[0]] + data.iloc[0: t + 1]  # pad with t0
            # res = []
            # Запишем текущий профит если сделка была
            # res.append(expit(dataOHLCV['Close'].iloc[t] - memory[-1] if memory else 0))
        profit = expit((self.df.Close.iloc[self.iloc] - self.broker.positions.get(self._name).price)
                       if self.broker.positions.get(self._name) else 0)
        block.append(profit)
            # Запишем цену последней сделки относительно цены последнего фрактала
        #res.append(expit(dataOHLCV['Close'].iloc[t] - data.iloc[-1]))
        close = expit(self.df.Close.iloc[self.iloc-1] - data[-1])
        block.append(close)
        # Запишем данные цен фракталов относительно цен соответсвующих им предидущих фркталов
        #     res = res + data.iloc[-(n_days - len(res) + 1):].diff().iloc[-(n_days - len(res)):].apply(expit).to_list()
        res = np.array([block])
        return res

class Data2(Data):
    ver = 'v03'
    description = '"Fractals paired" 1 level, not expit? diff with last close, MA, AT,State window_size=16'

    def _prepareDf(self):
        prepMa(self.df)
        prepAtr(self.df)
        # prepFractals(self.df)
        self.df['MIX'] = [OHLCVtoSeries(fractalsExp(win, 3)) for win in self.df.rolling(window=800)]
        # self.prepData = prepareData(self.df)

    def getState(self, n_days, memory, loc=None, *args, iloc=None, **kwargs):
        # def get_state3(data: list, t: int, n_days=10, memory=None, dataOHLCV=None):
        # data = OHLCVtoSeries(fractalsExp(dataOHLCV.iloc[:t], 3))
        return get_state3(['self.prepData'], iloc, n_days=n_days, memory=memory, dataOHLCV=self.df)

class Data3(Data1):
    """ Fractals pairs, with levels diff. Based on Data1 """
    ver = 'v03'
    def __init__(self, *args, **kwargs):
        self.window_size = kwargs['window_size']
        super().__init__(*args, **kwargs)
    def _prepareDf(self):
        super()._prepareDf()
        calculateFractalsPairs(self.df)
        self.fractalsValues = getStateFractalsValues(self.df)
        self.dfLevels = calcLevelsForEachInterval(self.df)
        for idx, row in self.df.iterrows():
            self.df.at[idx, 'nLevel'] = \
            sorted(self.dfLevels.loc[idx.floor(freq='D')], key=lambda x: abs(x - row.Close) / row.Close)[0]
        self.dff = pd.DataFrame([np.array(self.fractalsValues.loc[self.fractalsValues.index<idx].Close[-self.window_size+1:])-
                                 self.df.loc[idx].nLevel for idx,row in self.df.iterrows()],index=self.df.index)
    def getState(self, n_days, memory, loc=None, *args, iloc=None, **kwargs):
        """Returns an n-day state representation ending at time t
        """
        # data = OHLCVtoSeries(fractalsExp(self.df.iloc[:iloc], 3))
        data = self.dff.iloc[iloc].apply(expit).values
        # block = data.iloc[-n_days:]  # if d >= 0 else -d * [data.iloc[0]] + data.iloc[0: t + 1]  # pad with t0
        # res = []
        # Запишем текущий профит если сделка была
        profit = expit(self.df['Close'].iloc[iloc] - memory[-1]) if memory else 0.5

        # Запишем цену последней сделки относительно цены последнего фрактала
        # res.append(expit(self.df['Close'].iloc[iloc] - data.iloc[-1]))

        # Запишем данные цен фракталов относительно цен соответсвующих им предидущих фркталов
        # res = res + data.iloc[-(n_days - len(res) + 1):].diff().iloc[-(n_days - len(res)):].apply(expit).to_list()
        res = np.append(profit, data)
        return np.array([res])


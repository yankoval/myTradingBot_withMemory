"""
Script for training Stock Trading Bot.


Usage:
  train.py [--train-stock=<train-stock>] [--val-stock=<val-stock>] [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] [--log_dir=<log_dir>][--tfCounts=<tf-Counts>]  [--tik=<tiker>]
    [--tFrame=<tFrame>] [--dFrom=<dFrom>] [--dTo=<dTo>]
    [--vdFrom=<vdFrom>] [--vdTo=<vdTo>] [--trStrat=<trStrat>] [--trainId=<trainId>]
    [--dataPath=<dataPath>] [--evaluate_only=<evaluate_only>]
    
Options:
  --train-stock=<train-stock>       train data file ( if not set "--tik" used to read finamDB)
  --val-stock=<val-stock>           validate data file ( if not set "--tik" used to read finamDB)
  --strategy=<strategy>             Q-learning strategy to use for training the network. Options:
                                      `dqn` i.e. Vanilla DQN,
                                      `t-dqn` i.e. DQN with fixed target distribution,
                                      `double-dqn` i.e. DQN with separate network for value estimation. [default: t-dqn]
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
  --log_dir=<log_dir>               logging folder
  --tfCounts=<tf-Counts>            How math rows to get from data source [default: 0]
  --tik=<tiker>                     Tiker from Finam DB default None (read from CSV)
  --tFrame=<tFrame>                 Time frame (daily,hourly,minute,monthly,weekly) [default: daily]
  --dFrom=<dFrom>                   filter from Date default None 
  --dTo=<dTo>                       filter to Date default None 
  --vdFrom=<vdFrom>                 validate  filter from Date default None 
  --vdTo=<vdTo>                     validate  filter to Date default None 
  --trStrat=<trStrat>               trade strategy long, short or both  [default: long]
  --trainId=<trainId>               randomId of train filename suffix  [default: 0]
  --dataPath=<dataPath>             dataPath to DATASETs DB    [default: data]
  --evaluate_only=<evaluate_only>   evaluate pretrained models episodes range. --evaluate_only=1,2 episodes range(1,2)

"""

"""
Broker fee (Finam)
до 1 млн ₽
0,0354 % 
от 1 млн до 5 млн ₽
0,0295 %
от 5 млн до 10 млн ₽
0,0236 %
от 10 млн до 20 млн ₽
0,0177 %
от 20 млн до 50 млн ₽
0,01534 %
от 50 млн до 100 млн ₽
0,0118 %
от 100 млн ₽
0,00944 % 

Не менее 41,3 ₽ за исполненное поручение
"""
import logging
import coloredlogs

from docopt import docopt

from pathlib import Path
# from trading_bot.agent import Agent,AgentF, switch_k_backend_device
# from trading_bot.methodsCap import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    readData,
    prepareData,
    prepMa,prepAtr,prepFractals
)
from trading_bot.utils import Data3 as Data
from qbroker.broker import qbroker, AllInSizer
# from scipy.special import expit
from trading_bot.ops import OHLCVtoSeries,fractalsExp,get_state3

import numpy as np


def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, tfCounts = None
         ,tik=None                     #Tiker from Finam DB if none read from CSV
         ,tFrame='daily'               #Time frame (daily,hourly,minute,monthly,weekly)
         ,dFrom=None                   #train filter from Date default None 
         ,dTo=None                     #train filter to Date default None 
         ,vdFrom=None                   #validate filter from Date default None 
         ,vdTo=None                     #validate filter to Date default None 
         ,trStrat='long'                 #trade strategy long, short or both [default: long ]
         ,trainId='0'
         ,dataPath=r'D:/share/finam/data/'
         ,evaluate_only=None
         ,log_dir=None
         ):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    
    qt_v01 - based on qt from master trade, added current price in state, added reward for HOLD action
    qt_v02 - based on qt_v01 , reward is total profit function, removed reward for HOLD action
    qt_v03 - based on qt_v02 , Fractal level 1 totally compressed (removed all rows except fractals before train)
    qt_v04 - based on qt_v02 , Fractal level 3, train on every row, state calculated with current row + history
    qt_v05 - based on qt_v02 , Fractal level 3, train on every row, state calculated with current row + history
            + brokerFee param added to reward calculation compressed by fractals level 3
    qt_v06 - based on qt_v05 , changed reward policy to cumulative
    qt_v07 - based on qt_v06, state for training from data1 class
    qt_v08 - based on qt_v07, Model with 6 layers

    """

    if model_name in ['model_debug',None]:
        model_name = f'qt_v08_{strategy}_{trStrat}_{window_size}_{batch_size}_{tik}_{tFrame}_Agent_{Agent.ver}_Data_{Data.ver}'

    # Create a logger object.
    logger = logging.getLogger('train')
    coloredlogs.install(level=logging.DEBUG if debug else logging.INFO, fmt=f'%(asctime)s,%(name)s,%(levelname)s,{model_name}: %(message)s', logger=logger)

    log_dir = Path('.')/'logs' if log_dir is None else Path(log_dir)
    if not log_dir.exists():
        print('Log dir changed to current folder.')
        log_dir = Path('.')
    # Create a file handler object
    fh = logging.FileHandler(f'{(log_dir / (model_name+("_eval"if evaluate_only else "")))}.log')
    fh.setLevel(logging.DEBUG if debug else logging.INFO)

    # Create a ColoredFormatter to use as formatter for the FileHandler
    formatter = coloredlogs.ColoredFormatter(f'%(asctime)s,{model_name}_{trainId}: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    logger.info(f'log_dir: {log_dir}')
    logger.info(f'model_name: {model_name}')
    logger.info(f'dataPath: {dataPath}')
    
    if not evaluate_only:
        # if pretrained:
        #     if not Path('models/' + model_name).is_dir():
        #         raise RuntimeError(f'There is no models at {Path("models/" + model_name).absolute()}.')
        # agent = AgentF(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
        # train_dataOHLCV = readData(train_stock,dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame, dFrom=dFrom, dTo=dTo)
        # if train_dataOHLCV.empty:
        #     logger.error('Train dataset is empty.')
        #     return (-1)
        # logger.info(f'train data shape:{train_dataOHLCV.shape}, tfCounts: {tfCounts}, from:{train_dataOHLCV.iloc[0].name} '
        #             f'to:{train_dataOHLCV.iloc[tfCounts if train_dataOHLCV.shape[0]>tfCounts else (train_dataOHLCV.shape[0]-1)].name}.')
        # train_data = prepareData(train_dataOHLCV)
        # logger.info(f'Prepared train ver:0.1 data shape:{len(train_data)}')
        train_data = Data(train_stock if train_stock else dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame,
                              dFrom=dFrom,
                              dTo=dTo,
                          window_size=window_size,)
        if not train_data:
            logger.error('Train dataset is empty after preparation.')
            return (-1)
        logger.info(
            f'Train data shape:{train_data.df.shape}, tfCounts: {tfCounts}, from:{train_data.df.iloc[0].name} '
            f'to:{train_data.df.iloc[tfCounts if train_data.df.shape[0] > tfCounts else (train_data.df.shape[0] - 1)].name}.')
    # val_dataOHLCV = readData(val_stock, dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame, dFrom=vdFrom, dTo=vdTo)
    val_dataOHLCV = Data(val_stock if val_stock else dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame, dFrom=vdFrom,
               dTo=vdTo,window_size=window_size,)
    logger.info(f'Validation data data shape:{val_dataOHLCV.df.shape}, tfCounts: {tfCounts}, from:{val_dataOHLCV.df.iloc[0].name} '
                f'to:{val_dataOHLCV.df.iloc[tfCounts if val_dataOHLCV.df.shape[0] > tfCounts else (val_dataOHLCV.df.shape[0] - 1)].name}.')
    assert val_dataOHLCV.df.shape[0] > 800 , f'Shape:{val_dataOHLCV.df.shape} < 800.'
    val_dataOHLCV.next(iloc=800)
    valBro = qbroker(cash=1000)
    # valBro.set_cash(1000)
    valBro.setcommission(commission=0.001)
    val_dataOHLCV.setBroker(valBro)
    sizer = AllInSizer()
    val_dataOHLCV.setsizer(sizer)
    if val_dataOHLCV.df.empty:
        logger.error('Validate dataset is empty.') #8988 623 30 01 марг мих 370
        return (-1)
    #val_data = prepareData(val_dataOHLCV.df)
    # if not val_data:
    #     logger.error('Validate dataset is empty after preparation.')
    #     return (-1)

    logger.info(f'val data shape:{val_dataOHLCV.df.shape}, tfCounts: {tfCounts}, from:{val_dataOHLCV.df.iloc[0].name},'
                f'to:{val_dataOHLCV.df.iloc[tfCounts if val_dataOHLCV.df.shape[0]>tfCounts else (val_dataOHLCV.df.shape[0]-1)].name}.')
    if val_dataOHLCV.df.shape[0] <= 800 - 2:
        logger.error(f'Val shape:{val_dataOHLCV.df.shape[0]} less then {800 - 2}')
        raise
    initial_offset = 0.05

    # Evaluate models
    if evaluate_only:
        for i in range(evaluate_only[0],evaluate_only[1]) if len(evaluate_only)>1 else  evaluate_only:
            evaluate_only_turn = str(i)
            agent = Agent(None, strategy=strategy, pretrained=True, model_name=model_name + '_' + evaluate_only_turn)
            # Установим window_size из параметров загруженной модели.
            if window_size != agent.state_size:
                logger.error(f'window size parameter not match to loadad model. Set window size from loaded model!')
                return (-1)
                # window_size = agent.state_size
            val_result, history, maxDrawdownAbs = evaluate_model(agent, val_dataOHLCV, window_size, debug,startFrom=800)
            show_train_result((1,2,3,4), val_result, initial_offset, history=history, df=val_dataOHLCV
                              ,maxDrawdownAbs=maxDrawdownAbs,modelName=model_name+'_'+evaluate_only_turn,
                              )
            logger.info(f'Option evaluate_only is: {evaluate_only_turn}!!!!!!!!!!!!!!!!')
        return


    # Train model
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    if not pretrained:
        agent.save(0)
    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
            batch_size=batch_size, window_size=window_size,
            rewardFunc = 'calcRewardLine',
            trStrat=trStrat,
            #dataOHLCV = train_dataOHLCV,
            brokerFee=0.001
            )
        val_result, history, maxDrawdownAbs = evaluate_model(agent, val_dataOHLCV, window_size, debug,startFrom=800
                                                            #, dataOHLCV=val_dataOHLCV.df
                                                            #, brokerFee=0.001
                                                             )
        show_train_result(train_result, val_result, initial_offset, history=history, df=val_dataOHLCV
                          , modelName=model_name, maxDrawdownAbs=maxDrawdownAbs)


if __name__ == "__main__":
    print(Path('.').absolute())
    args = docopt(__doc__)
    print(args)

    train_stock = args["--train-stock"]
    val_stock = args["--val-stock"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]
    tfCounts = int(args["--tfCounts"]) if int(args["--tfCounts"])!=0 else None
    tik = args["--tik"]
    tFrame = args["--tFrame"]
    dFrom = args["--dFrom"]
    dTo = args["--dTo"]
    vdFrom = args["--vdFrom"]
    vdTo = args["--vdTo"]
    trStrat = args["--trStrat"]
    trainId = args["--trainId"]
    dataPath = args["--dataPath"]
    evaluate_only=list(map(int,args["--evaluate_only"].split(','))) if isinstance(args["--evaluate_only"],str) else args["--evaluate_only"]
    log_dir = args["--log_dir"]
    # Parameters checks

    # Check path to data:
    if not Path(dataPath).is_dir():
        raise RuntimeError(f'There is no data dir at {Path(dataPath).absolute()}.')

    # Check strategy
    if not strategy in ["t-dqn", "double-dqn", "dqn"]:
        raise RuntimeError(f'There is no data dir at {Path(dataPath).absolute()}.')

    from trading_bot.agent import switch_k_backend_device
    from trading_bot.agent import AgentF as Agent
    from trading_bot.methodsCap import train_model, evaluate_model

    switch_k_backend_device()




    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug,tfCounts=tfCounts,
             tik=tik,tFrame=tFrame,dFrom=dFrom,dTo=dTo,vdFrom=vdFrom,
             vdTo=vdTo,trStrat=trStrat, trainId=trainId, dataPath=dataPath
             ,evaluate_only=evaluate_only,log_dir=log_dir)
    except KeyboardInterrupt:
        print("Aborted!")

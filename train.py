"""
Script for training Stock Trading Bot.


Usage:
  train.py [--train-stock=<train-stock>] [--val-stock=<val-stock>] [--agentClass=<agentClass>] [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug] [--log_dir=<log_dir>][--tfCounts=<tf-Counts>]  [--tik=<tiker>]
    [--tFrame=<tFrame>] [--dFrom=<dFrom>] [--dTo=<dTo>]
    [--vdFrom=<vdFrom>] [--vdTo=<vdTo>] [--start_from=<start_from>] [--trStrat=<trStrat>] [--trainId=<trainId>]
    [--dataPath=<dataPath>] [--dataClass=<dataClass>] [--evaluate_only=<evaluate_only>]
    
Options:
  --train-stock=<train-stock>       train data file ( if not set "--tik" used to read finamDB)
  --val-stock=<val-stock>           validate data file ( if not set "--tik" used to read finamDB)
  --agentClass=<agentClass>           agent Class name, default basic agent for t-dqn [default: Agent]
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
  --start_from=<start_from>         Starting train index  [default: "0"]
  --trStrat=<trStrat>               trade strategy long, short or both  [default: long]
  --trainId=<trainId>               randomId of train filename suffix  [default: 0]
  --dataPath=<dataPath>             dataPath to DATASETs DB    [default: data]
  --dataClass=<dataClass>           data Class name, default basic OHLCV [default: Data]
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

--trStrat=Long
--tik=MXZ4
--strategy=t-dqn
--window-size=15
--batch-size=32
--episode-count=50
--tFrame=min
--dFrom=2024.09.30
--dTo=2024.12.04
--vdFrom=2024.08.01
--vdTo=2024.10.30
--start_from=35000
--trainId=778
--debug
--dataClass=DataV303
--agentClass=AgentF
--pretrained

"""
import logging
import logging.config
import coloredlogs
import traceback

from docopt import docopt

from pathlib import Path
from trading_bot.utils import show_train_result
from qbroker.broker import qbroker, AllInSizer


def main(train_stock, val_stock, window_size, batch_size, ep_count
         ,strategy="t-dqn", model_name="model_debug"
         ,agentClass = None
         ,pretrained=False
         ,dataClass=None
         ,tfCounts = None
         ,tik=None                     #Tiker from Finam DB if none read from CSV
         ,tFrame='daily'               #Time frame (daily,hourly,minute,monthly,weekly)
         ,dFrom=None                   #train filter from Date default None 
         ,dTo=None                     #train filter to Date default None 
         ,vdFrom=None                   #validate filter from Date default None 
         ,vdTo=None                     #validate filter to Date default None
         ,start_from=1
         ,trStrat='long'                 #trade strategy long, short or both [default: long ]
         ,trainId='0'
         ,dataPath=r'D:/share/finam/data/'
         ,evaluate_only=None
         ,log_dir=None
         ,debug=False
         ,
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
    # Create a logger object.
    logging.config.fileConfig(Path('logging.conf').absolute(), disable_existing_loggers=False)
    logger = logging.getLogger(__name__)

    # import dynamic classes
    import importlib
    Data = importlib.import_module('trading_bot.utils').__dict__[dataClass]
    Agent = importlib.import_module('trading_bot.agent').__dict__[agentClass]

    if model_name in ['model_debug',None]:
        model_name = f'qt_v09_{strategy}_{trStrat}_{window_size}_{batch_size}_{tik}_{tFrame}_Agent_{agentClass}_Data_{dataClass}'

    coloredlogs.install(fmt=f'%(asctime)s,%(name)s,%(levelname)s,{model_name}: %(message)s', logger=logger)

    log_dir = Path('.')/'logs' if log_dir is None else Path(log_dir)
    if not log_dir.exists():
        print('Log dir changed to current folder.')
        log_dir = Path('.')
    # Create a file handler object
    fh = logging.FileHandler(f'{(log_dir / (model_name+("_eval"if evaluate_only else "")))}.log')
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    # Create a ColoredFormatter to use as formatter for the FileHandler
    formatter = coloredlogs.ColoredFormatter(f'%(asctime)s,%(levelname)s,{model_name}_{trainId}: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # filter log modules matplotlib.category
    logger_urllib3 = logging.getLogger('urllib3')
    logger_urllib3.setLevel(logging.ERROR)
    logger_urllib3 = logging.getLogger('matplotlib.category')
    logger_urllib3.setLevel(logging.ERROR)

    logger.info(f'log_dir: {log_dir}')
    logger.info(f'model_name: {model_name}')
    logger.info(f'dataPath: {dataPath}, dataClass: {dataClass}')


    if not evaluate_only:
        train_data = Data(train_stock if train_stock else dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame,
                              dFrom=dFrom,
                              dTo=dTo,
                          window_size=window_size,start_from=start_from)
        if not train_data:
            logger.error('Train dataset is empty after preparation.')
            return (-1)
        logger.info(
            f'Train data shape:{train_data.df.shape},'
             f' from:{train_data.df.index[0]}, start_from:{start_from},'
            f' to:{train_data.df.index[-1]}.')
    # val_dataOHLCV = readData(val_stock, dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame, dFrom=vdFrom, dTo=vdTo)
    val_dataOHLCV = Data(val_stock if val_stock else dataPath, tfCounts=tfCounts, tik=tik, tFrame=tFrame, dFrom=vdFrom,
               dTo=vdTo,window_size=window_size,start_from=start_from,)
    logger.info(f'Validation data data shape:{val_dataOHLCV.df.shape}, '
                f' from:{val_dataOHLCV.df.index[0]}, start_from:{start_from},'
                f' to:{val_dataOHLCV.df.index[-1]}.')
    assert val_dataOHLCV.df.shape[0] > 800 , f'Shape:{val_dataOHLCV.df.shape} < 800.'
    if val_dataOHLCV.df.empty:
        logger.error('Validate dataset is empty.') #8988 623 30 01 марг мих 370
        return (-1)

    initial_offset = 0.05

    # Evaluate models
    if evaluate_only:
        for i in range(evaluate_only[0],evaluate_only[1]) if len(evaluate_only)>1 else  evaluate_only:
            try:
                agent = Agent(None, strategy=strategy, pretrained=True,
                            modelPath=(Path.cwd() /'models').absolute().__str__(),
                            model_name=f"{model_name}_episode_{i}"
                              )
                # Установим window_size из параметров загруженной модели.
                if window_size != agent.state_size:
                    logger.error(f'window size parameter not match to loadad model. Set window size from loaded model!')
                    return (-1)
                    # window_size = agent.state_size
                start_from = start_from if type(start_from) is int else val_dataOHLCV.df.index.get_loc(start_from).start
                val_dataOHLCV.next(iloc=start_from)

                valBro = qbroker(cash=1000000)
                # valBro.set_cash(1000)
                valBro.setcommission(commission=0.0001, name=tik)
                val_dataOHLCV.setBroker(valBro)
                sizer = AllInSizer()
                val_dataOHLCV.setsizer(sizer)

                val_result, history, maxDrawdownAbs = evaluate_model(agent, val_dataOHLCV, window_size, debug
                                                                     ,start_from=start_from,
                                                                     logger=logger)
                show_train_result((1,2,3,4), val_result, initial_offset, history=history, data=val_dataOHLCV
                                  ,maxDrawdownAbs=maxDrawdownAbs,modelName=model_name+'_'+str(i), start_from=start_from
                                  )
                logger.info(f'Option evaluate_only is: {i}.')
            except Exception as e:
                logger.error(f'evaluate_only: on{i} step generate error: {e}. {traceback.format_exc()}')
                traceback.print_exc()
        return


    # Train model
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    # if not pretrained:
    #     agent.save(0)
    for episode in range(agent.episode+1, ep_count + 1):
        start_from = start_from if type(start_from) is int else val_dataOHLCV.df.index.get_loc(start_from).start
        train_data.next(iloc=start_from)
        bro = qbroker(cash=1000000)
        bro.setcommission(commission=0.0001, name=tik)
        train_data.setBroker(bro)
        sizer = AllInSizer()
        train_data.setsizer(sizer)
        train_result = train_model(agent, train_data, episode=episode, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size,
                                   reward_func='calcRewardLine',
                                   tr_strat=trStrat,
                                   start_from=start_from,
                                   broker_fee=0.0001
                                   )
        try:
            if type(start_from) is int:
                val_dataOHLCV.next(iloc=start_from)
            else:
                val_dataOHLCV.next(loc=start_from)
            valBro = qbroker(cash=1000000)
            valBro.setcommission(commission=0.0001, name=tik)
            val_dataOHLCV.setBroker(valBro)
            sizer = AllInSizer()
            val_dataOHLCV.setsizer(sizer)

            val_result, history, maxDrawdownAbs = evaluate_model(agent, val_dataOHLCV, window_size
                                                                 , debug
                                                                 , logger=logger
                                                                 , start_from=start_from
                                                                 #, brokerFee=0.001
                                                                 )
            with open(f'{(log_dir / (model_name + (f"_episode_{ep_count}" )))}.hist','w') as f:
                for h in history:
                    f.write(str(h))
            show_train_result(train_result, val_result, initial_offset, history=history, data=val_dataOHLCV
                              , modelName=model_name, maxDrawdownAbs=maxDrawdownAbs, start_from=start_from)
        except Exception as e:
            logger.error(f'evaluate_model: {e}, {traceback.format_exc()}')
            # print(traceback.print_exc())



if __name__ == "__main__":
    args = docopt(__doc__)

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
    start_from = args["--start_from"]
    trStrat = args["--trStrat"]
    trainId = args["--trainId"]
    dataPath = args["--dataPath"]
    evaluate_only=list(map(int,args["--evaluate_only"].split(','))) if isinstance(args["--evaluate_only"],str) else args["--evaluate_only"]
    log_dir = args["--log_dir"]
    dataClass = args["--dataClass"]
    agentClass = args["--agentClass"]
    # Parameters checks

    # Check path to data:
    if not Path(dataPath).is_dir():
        raise RuntimeError(f'There is no data dir at {Path(dataPath).absolute()}.')

    # Check strategy
    if not strategy in ["t-dqn", "double-dqn", "dqn"]:
        raise RuntimeError(f'There is no data dir at {Path(dataPath).absolute()}.')

    from trading_bot.agent import switch_k_backend_device
    # from trading_bot.agent import AgentF as Agent
    from trading_bot.methodsCap import train_model, evaluate_model

    # switch_k_backend_device()




    try:
        main(train_stock, val_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name
             ,agentClass=agentClass
             ,pretrained=pretrained, debug=debug,tfCounts=tfCounts,
             tik=tik,tFrame=tFrame,dFrom=dFrom,dTo=dTo,vdFrom=vdFrom,
             vdTo=vdTo,trStrat=trStrat, trainId=trainId, dataPath=dataPath
             ,evaluate_only=evaluate_only,log_dir=log_dir, start_from=start_from
             ,dataClass=dataClass)
    except KeyboardInterrupt:
        print("Aborted!")

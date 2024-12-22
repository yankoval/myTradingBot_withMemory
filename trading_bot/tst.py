# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trading_bot.ops import calcRewardLine
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device,
    readData,
    prepareData
)

# %%
df = readData(None,tik='GAZP',tFrame='minute', dFrom='2021.03.01',dTo='2021.03.10',tfCounts=600)

# %%
p1 = df
p2 = df[df.High==df.Close]
# %%
def pltHist(p1,hist,fName='tmp.jpg'):
    b = [[r[0] for r in hist if r[2] == 'BUY'], [r[1] for r in hist if r[2] == 'BUY']]
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax1.plot(p1.index, p1.High, 'g')
    ax1.plot(p1.index, p1.Low,'y')
    #ax1.scatter(p1.iloc[b[0]].index, b[1],marker='o',s=300)
    ax1.scatter(p1.iloc[b[0]].index, b[1],marker='o',s=300)
    if fName:
        plt.savefig(fName)
        print(fName)
    plt.show()


# %%
show_train_result([1,2,3,4,5],3,11, history=[(500, 285.55, 'BUY'), (501, 285.7, 'SELL'), (502, 285.75, 'BUY'), (503, 285.57, 'SELL'), (504, 285.6, 'BUY'), (505, 285.73, 'SELL'), (506, 285.69, 'BUY')]
                  , df=df, modelName='tst')
#pltHist(df,[(500, 285.55, 'BUY'), (501, 285.7, 'SELL'), (502, 285.75, 'BUY'), (503, 285.57, 'SELL'), (504, 285.6, 'BUY'), (505, 285.73, 'SELL'), (506, 285.69, 'BUY')])

# %%time

np.array([df.iloc[-10:].Close])
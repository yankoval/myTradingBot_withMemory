#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
###############################################################################
#
# Copyright (C) 2015-2020 Ivan Kiselev
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
"""


# from . import fillers as fillers
# from . import fillers as filler
from qbroker.metabase import with_metaclass
from qbroker.metabase import MetaParams
import collections

# class Sizer(with_metaclass(MetaParams)):

class CommInfoBase(with_metaclass(MetaParams)):
    '''Base Class for the Commission Schemes.

    Params:

      - ``commission`` (def: ``0.0``): base commission value in percentage or
        monetary units

      - ``mult`` (def ``1.0``): multiplier applied to the asset for
        value/profit

      object

    '''

    COMM_PERC, COMM_FIXED = range(2)

    params = (
        ('commission', 0.0), ('mult', 1.0), ('margin', None),
        ('commtype', None),
        ('stocklike', False),
        ('percabs', False),
        ('interest', 0.0),
        ('interest_long', False),
        ('leverage', 1.0),
        ('automargin', False),
    )

    def __init__(self):
        super(CommInfoBase, self).__init__()

        self._stocklike = self.p.stocklike
        self._commtype = self.p.commtype

        # The intial block checks for the behavior of the original
        # CommissionInfo in which the commission scheme (perc/fixed) was
        # determined by parameter "margin" evaluating to False/True
        # If the parameter "commtype" is None, this behavior is emulated
        # else, the parameter values are used

        if self._commtype is None:  # original CommissionInfo behavior applies
            if self.p.margin:
                self._stocklike = False
                self._commtype = self.COMM_FIXED
            else:
                self._stocklike = True
                self._commtype = self.COMM_PERC

        if not self._stocklike and not self.p.margin:
            self.p.margin = 1.0  # avoid having None/0

        if self._commtype == self.COMM_PERC and not self.p.percabs:
            self.p.commission /= 100.0

        self._creditrate = self.p.interest / 365.0

    @property
    def margin(self):
        return self.p.margin

    @property
    def stocklike(self):
        return self._stocklike

    def get_margin(self, price):
        '''Returns the actual margin/guarantees needed for a single item of the
        asset at the given price. The default implementation has this policy:

          - Use param ``margin`` if param ``automargin`` evaluates to ``False``

          - Use param ``mult`` * ``price`` if ``automargin < 0``

          - Use param ``automargin`` * ``price`` if ``automargin > 0``
        '''
        if not self.p.automargin:
            return self.p.margin

        elif self.p.automargin < 0:
            return price * self.p.mult

        return price * self.p.automargin  # int/float expected


class MetaBroker(MetaParams):
    def __init__(cls, name, bases, dct):
        '''
        Class has already been created ... fill missing methods if needed be
        '''
        # Initialize the class
        super(MetaBroker, cls).__init__(name, bases, dct)
        translations = {
            'get_cash': 'getcash',
            'get_value': 'getvalue',
        }

        for attr, trans in translations.items():
            if not hasattr(cls, attr):
                setattr(cls, name, getattr(cls, trans))


class Position():
    def __init__(self, *args, **kwargs):
        self.price = .0
        self.size = 0
        self.date = None

    def reg(self, price:float, size:int, date):
        if not price:
            raise ('price not given')
        if self.size + size ==0:
            self.price = .0
            self.size = 0
            self.date = None
        else:
            try:
                if size:
                    size = size
                else:
                    return
                price = price if price else 0
                self.price = (self.size * self.price + size*price)/(self.size+size)
                self.size += size
                self.date = date
            except Exception as e:
                print(e)
                raise
    def __repr__(self):
        return f'{self.size}:{self.price:.2f} - {self.date}'

class TradeBase(with_metaclass(MetaParams)):
    params = (
        ('type','BUYSLTP'),
        ('priceUnit','ATR'),
        ('price',-1),
        ('size', 1),
        ('slUnit','ATR'),
        ('sl', -2),
        ('tpUnit', '%'),
        ('tp', 1),
    )
    def __init__(self,*args,**kwargs):
        super(TradeBase, self).__init__()

        self.init()

    def init(self):
        for attr in [l for l in self.p.__dir__() if l[0]!='_']:
            # if not hasattr(self, attr):
            setattr(self, attr, self.p._get(attr))

class Sizer(with_metaclass(MetaParams, object)):
    '''This is the base class for *Sizers*. Any *sizer* should subclass this
    and override the ``_getsizing`` method

    Member Attribs:

      - ``strategy``: will be set by the strategy in which the sizer is working

        Gives access to the entire api of the strategy, for example if the
        actual data position would be needed in ``_getsizing``::

           position = self.strategy.getposition(data)

      - ``broker``: will be set by the strategy in which the sizer is working

        Gives access to information some complex sizers may need like portfolio
        value, ..
    '''
    strategy = None
    broker = None

    def getsizing(self, data, isbuy):
        comminfo = self.broker.getcommissioninfo(data)
        return self._getsizing(comminfo, self.broker.getcash(), data, isbuy)

    def _getsizing(self, comminfo, cash, data, isbuy):
        '''This method has to be overriden by subclasses of Sizer to provide
        the sizing functionality

        Params:
          - ``comminfo``: The CommissionInfo instance that contains
            information about the commission for the data and allows
            calculation of position value, operation cost, commision for the
            operation

          - ``cash``: current available cash in the *broker*

          - ``data``: target of the operation

          - ``isbuy``: will be ``True`` for *buy* operations and ``False``
            for *sell* operations

        The method has to return the actual size (an int) to be executed. If
        ``0`` is returned nothing will be executed.

        The absolute value of the returned value will be used

        '''
        raise NotImplementedError

    def set(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker
class PercentSizer(Sizer):
    '''This sizer return percents of available cash

    Params:
      - ``percents`` (default: ``20``)
    '''

    params = (
        ('percents', 20),
        ('retint', False),  # return an int size or rather the float value
    )

    def __init__(self):
        pass

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        if not position.size:
            size = int(cash / data.df.Close.iloc[data.iloc] * (self.params.percents / 100))
        else:
            size = abs(position.size)

        if self.p.retint:
            size = int(size)

        return size


class AllInSizer(PercentSizer):
    '''This sizer return all available cash of broker

     Params:
       - ``percents`` (default: ``100``)
     '''
    params = (
        ('percents', 100),
    )


class BrokerBase(with_metaclass(MetaBroker, object)): #with_metaclass(MetaBroker, object)
    params = (
        ('commission', CommInfoBase()),
    )

    def __init__(self):
        self.comminfo = dict()
        self.data = None
        self.datas = []
        self.init()

    def init(self):
        # called from init and from start
        if None not in self.comminfo:
            self.comminfo = dict({None: self.params.commission})

    def start(self):
        self.init()

    def stop(self):
        pass

    def add_order_history(self, orders, notify=False):
        '''Add order history. See cerebro for details'''
        raise NotImplementedError

    def set_fund_history(self, fund):
        '''Add fund history. See cerebro for details'''
        raise NotImplementedError

    def getcommissioninfo(self, data):
        '''Retrieves the ``CommissionInfo`` scheme associated with the given
        ``data``'''
        if data._name in self.comminfo:
            return self.comminfo[data._name]

        return self.comminfo[None]

    def setcommission(self,
                      commission=0.001, margin=None, mult=1.0,
                      commtype=None, percabs=False, stocklike=False,
                      interest=0.0, interest_long=False, leverage=1.0,
                      automargin=False,
                      name=None):

        '''This method sets a `` CommissionInfo`` object for assets managed in
        the broker with the parameters. Consult the reference for
        ``CommInfoBase``

        If name is ``None``, this will be the default for assets for which no
        other ``CommissionInfo`` scheme can be found
        '''

        comm = CommInfoBase(commission=commission, margin=margin, mult=mult,
                            commtype=commtype, stocklike=stocklike,
                            percabs=percabs,
                            interest=interest, interest_long=interest_long,
                            leverage=leverage, automargin=automargin)
        self.comminfo[name] = comm

    def addcommissioninfo(self, comminfo, name=None):
        '''Adds a ``CommissionInfo`` object that will be the default for all assets if
        ``name`` is ``None``'''
        self.comminfo[name] = comminfo

    def getcash(self):
        raise NotImplementedError

    def getvalue(self, datas:list=None):
        datas = datas if datas else self.datas
        datas = datas if hasattr(datas, '__iter__') else [datas]
        res = 0
        for data in datas:
            tik = self.getposition(data)
            price = data.df.Close.iloc[data.iloc]
            if tik.size<0: res += -tik.size * tik.price -tik.size * (tik.price - price)
            else: res += tik.size * price
        return res

    def get_fundshares(self):
        '''Returns the current number of shares in the fund-like mode'''
        return 1.0  # the abstract mode has only 1 share

    fundshares = property(get_fundshares)

    def get_fundvalue(self):
        return self.getvalue()

    fundvalue = property(get_fundvalue)

    def set_fundmode(self, fundmode, fundstartval=None):
        '''Set the actual fundmode (True or False)

        If the argument fundstartval is not ``None``, it will used
        '''
        pass  # do nothing, not all brokers can support this

    def get_fundmode(self):
        '''Returns the actual fundmode (True or False)'''
        return False

    fundmode = property(get_fundmode, set_fundmode)

    def getposition(self, data=None):
        data = data if data else self.data
        return self.positions.get(data._name)

    def submit(self, order):
        raise NotImplementedError

    def cancel(self, order):
        raise NotImplementedError

    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):

        raise NotImplementedError

    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):

        raise NotImplementedError

    def next(self):
        pass

# __all__ = ['BrokerBase', 'fillers', 'filler']

class qbroker(BrokerBase): #BrokerBase
    params = (
        ('cash', 10000.0),
        ('checksubmit', True),
        ('eosbar', False),
        ('filler', None),
        # slippage options
        ('slip_perc', 0.0),
        ('slip_fixed', 0.0),
        ('slip_open', False),
        ('slip_match', True),
        ('slip_limit', True),
        ('slip_out', False),
        ('coc', False),
        ('coo', False),
        ('int2pnl', True),
        ('shortcash', True),
        ('fundstartval', 100.0),
        ('fundmode', False),
    )

    def __init__(self):
        super(qbroker, self).__init__()
        self._userhist = []
        self._fundhist = []
        # share_value, net asset value
        self._fhistlast = [float('NaN'), float('NaN')]
        self.dealsHist = []


    def init(self):
        super(qbroker, self).init()
        self.startingcash = self.cash = self.p.cash
        self._value = self.cash
        self._valuemkt = 0.0  # no open position

        self._valuelever = 0.0  # no open position
        self._valuemktlever = 0.0  # no open position

        self._leverage = 1.0  # initially nothing is open
        self._unrealized = 0.0  # no open position

        self.orders = list()  # will only be appending
        self.pending = collections.deque()  # popleft and append(right)
        self._toactivate = collections.deque()  # to activate in next cycle

        self.positions = collections.defaultdict(Position)
        self.d_credit = collections.defaultdict(float)  # credit per data
        self.notifs = collections.deque()

        self.submitted = collections.deque()

        # to keep dependent orders if needed
        self._pchildren = collections.defaultdict(collections.deque)

        self._ocos = dict()
        self._ocol = collections.defaultdict(list)

        self._fundval = self.p.fundstartval
        self._fundshares = self.p.cash / self._fundval
        self._cash_addition = collections.deque()
    def get_cash(self):
        '''Returns the current cash (alias: ``getcash``)'''
        return self.cash

    getcash = get_cash

    def set_cash(self, cash):
        '''Sets the cash parameter (alias: ``setcash``)'''
        self.startingcash = self.cash = cash
        self._value = cash
    # Проверяет достаточность средств, возвращает количество наличных остающихся после сделки. Если их >=0 то ОК
    def kKontrol(self,size,price,data=None):
        data = data if data else self.data
        pos = self.getposition(data)
        need_many_to_do =abs(pos.size + size) * price
        cur_val = self.getcash() + self.getvalue(self.data)
        return cur_val - need_many_to_do
    def buy(self, owner=None, data=None, size=None, price=None, plimit='Close',
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):
        data = data if data else self.data
        size = size if size is not None else data._sizer.getsizing(data, isbuy=True)
        if plimit == 'Close':
            price = data.df.iloc[data.iloc-1].Close
        else:
            raise NotImplementedError
        pos = self.getposition(data)
        # Подсчитываем хватит ли средств
        cash = self.kKontrol(size,price)
        if cash < 0:
            return False
        self.positions[data._name].reg(price,size,data.loc)
        self.cash = cash
        self.dealsHist.append(dict(iloc=self.data.iloc, price=price, type="BUY", total_profit=0, maxDrawdownAbs=0,loc=self.data.loc,))
        return True

    def sell(self, owner=None, data=None, size=None, price=None, plimit='Close',
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):
        data = data if data else self.data
        size = size if size is not None else data._sizer.getsizing(data, isbuy=False)
        if plimit == 'Close':
            price = data.df.iloc[data.iloc-1].Close
        else:
            raise NotImplementedError
        pos = self.getposition(data)
        # Подсчитываем хватит ли средств
        cash = self.kKontrol(-size,price)
        if cash < 0:
            return False
        self.positions[data._name].reg(price, -size,data.loc)
        self.cash = cash
        self.dealsHist.append(
            dict(iloc=self.data.iloc, price=price, type="SELL", total_profit=0, maxDrawdownAbs=0, loc=self.data.loc, ))
        return True
    def calcTrade(self,data,order):
        df = data.getDf()
        if order.type == 'BUYSLTP':
            if order.priceUnit == 'ATR':
                price = df.Close.iloc[0] + df.ATR.iloc[0] * order.price
            buyIdx = df[df.Low<price].index
            if buyIdx.empty: return 1
            buyIdx = buyIdx[0]
            cl = df.loc[buyIdx:]
            if order.tpUnit == '%':
                tpActPrice = price * (1 + order.tp / 100)
            tp = cl[df.High>=tpActPrice].index
            if order.slUnit == 'ATR':
                slActPrice = cl.Close.iloc[0] + cl.ATR.iloc[0] * order.sl
            sl = cl[df.Low<=slActPrice].index
            if any(sl) and any(tp):
                if tp[0]<=sl[0]: sl = [False]
                else: tp = [False]
            if any(sl):
                return slActPrice/price
            if any(tp):
                return 1+order.tp / 100
            return cl.Close.iloc[-1] / price

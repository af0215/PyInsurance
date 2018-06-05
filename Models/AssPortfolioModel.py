import datetime as dt

from Securities.AssBond import AssAsset, AssBondFixRate, AssBondFloater
from Models.AssModel import AssModelBase, NoDefaultCreditModel
from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager


class AssPortfolioModelBase(object):
    def __init__(self,
                 instruments,
                 quantities,
                 credit_models=None):
        """
        a simple collection of assets. this should be in the replacable with model_iter
        so it should provide all interfaces outside world needs:
        .next: roll each instrument forwrdTing

        instruments: are of type AssAsset
        quantities: should be of float
        credit_models: so far

        Each model inside can have its own credit model.
        Notice I dont define a CF here as it will be handled by Curve Aggregator
        NOTE THAT STRAIGHT SUM UP OF CF FROM EACH INSTRUMENT IS WRONG, SINCE THOSE ARE WITHOUT CREDIT (i.e. not scaled by survival)
        """
        # so far bond base is the only support asset in this simple model
        assert len(instruments) == len(quantities), "you have %s instruments, but %s quantities" % (len(instruments), len(quantities))
        assert all(isinstance(inst, AssAsset) for inst in instruments), "not all instruments are derived from AssModelBase"

        if credit_models:
            assert len(credit_models) == len(instruments), "you should specify one credit model per instrument"
        else:
            self._credit_models = [NoDefaultCreditModel()] * len(instruments)  # if default is none credit model for all

        self._instruments = instruments
        self._quantities = quantities
        self._asset_models = [ AssModelBase(inst, credit_model) for inst, credit_model in zip(self._instruments, self._credit_models)]

        self._model_iters = None # initialize to be None, need a pricing_date to create; creation in "create_iterator"

    def create_iterator(self, pricing_date):
        self._model_iters = [ model.create_iterator(pricing_date) for model in self._asset_models]


    @property
    def instruments(self):
        return self._instruments

    @property
    def quantities(self):
        return self._quantities

    @property
    def asset_models(self):
        return self._asset_models

    @property
    def model_iters(self):
        return self._model_iters

    def __str__(self):

        """
        the print is not perfect, each row should be of instrument, model, quantity
        """
        s = ''
        for inst, model, q in zip(self._instruments, self._asset_models, self._quantities):
            s += '%s, %s: %s\n' % (inst, model, q)

        return s


    def next(self, _next_date, _year_frac):
        """
            just calls each instrument and roll forward
        """

        if self._model_iters is None:
            raise Exception('create model iterator first from this portfolio')

        for itr in self._model_iters:
            itr.next(_next_date, _year_frac)

    @property
    def date(self):
        if self._model_iters is None:
            raise Exception('create model iterator first from this portfolio')

        return self._model_iters[0].date



def main():
    from lib.utils import dcf_act_act
    from Managers.MarketDataManager import MARKET_DATA_MANAGER

    MARKET_DATA_MANAGER.reset()
    MARKET_DATA_MANAGER.setup(dt.date(2014,4,15))
    # Setup a bond
    asset1 = AssBondFixRate(face=100,
                               coupon_rate=0.05,
                               frequency=4,
                               issue_date=dt.date(2014, 4, 15),
                               expiration_date=dt.date(2015, 4, 15),
                               dcf=dcf_act_act,
                               pricing_model=None,
                               name='Sample Bond 1',
                            )

    # or a "float rate bond" but provided with a scenario generator gives you flat rate
    # a LIBOR 3M +50 bps floater
    asset2 = AssBondFloater(rate_index_name='LIBOR_3M', spread=0.005)

    pricing_date = dt.date(2014, 4, 15)

    # create a portfolio
    port = AssPortfolioModelBase([asset1,asset2], [1,1])
    port.create_iterator(dt.date(2014, 4, 15))
    print(port)

    metrics = ['Date', 'CF_In:Interest', 'CF_In:Principal']

    crv_aggregator = create_curve_aggregator(metrics)

    params = {'pricing date': pricing_date, 'periods': 60, 'frequency': 'MS'}
    proj_mgr = ProjectionManager(crv_aggregator, port, **params)
    proj_mgr.run()

    df = crv_aggregator.to_dataframe()
    print(df)


if __name__ == '__main__':
    from Models.AssPortfolioModel import AssPortfolioModelBase
    main()
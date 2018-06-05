"""VA model"""

import datetime as dt

import pandas as pd
import numpy as np

from Models.InsModelFA import InsModelFA, InsModelFAIter
from Account.InsAcct import InsAcct
import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.utils import extract_strict
from Models.InsMortModel import InsMortModel
from Models.InsLapseModel import SurrenderCharge, LapseDynamic
from lib.insurance import InsStepFunc, linear_comp_bounded
from Infra.CurveAggregator import create_curve_aggregator
from Managers.ProjectionManager import ProjectionManager
import Infra.IndexProvider as ip
from Managers.ScenarioManager import EqBSEngine, ScenarioGenerator
from lib.constants import BDAYS_PER_YEAR
from Products.InsProduct import InsProduct


OUTPUT_PATH = 'Output/VA'


class InsModelVA(InsModelFA):
    pass


class InsModelVAIter(InsModelFAIter):
    pass


def main():
    from Managers.MarketDataManager import MARKET_DATA_MANAGER

    raw_input = {"Acct Value": 1344581.6,
                 "Attained Age": 52.8,
                 "ID": "000001",
                 "Issue Age": 45.1,
                 "Issue Date": dt.date(2005, 6, 22),
                 "Initial Date": dt.date(2013, 2, 1),
                 "Maturity Age": 90,
                 "Population": 1,
                 "Riders": dict({}),
                 "ROP Amount": 1038872.0,
                 "Gender": "F",
                 "RPB": 1038872.0,
                 "Free Withdrawal Rate": 0.1,
                 "Asset Names": ["Fund A", "Fund B"],
                 "Asset Values": [1344581.6/2, 1344581.6/2]}

    # For now, we assume the init_date is month begin
    step_per_year = 12
    periods = 360
    init_date = dt.date(2013, 2, 1)
    pricing_date = init_date
    # Set up the investment index
    #credit_rider = isr.InsCreditRateFixed(credit_rate)

    # set up the mutual fund return index
    init_df = [pd.TimeSeries(data=[100], index=[init_date], name='stock A'),
               pd.TimeSeries(data=[100], index=[init_date], name='stock B')
    ]
    eq_index = [ip.IndexProvider(init_df[0], 'stock A'), ip.IndexProvider(init_df[1], 'stock B')]
    sim_engine = EqBSEngine(np.array([0.02, 0.02]), np.array([0.2, 0.25]), corr=np.array([[1., 0.3], [0.3, 1.]]))
    simulator = ScenarioGenerator(eq_index, sim_engine, **{'max_time_step': 5. / BDAYS_PER_YEAR})

    MARKET_DATA_MANAGER.reset()
    MARKET_DATA_MANAGER.setup(init_date)
    MARKET_DATA_MANAGER.index_table['stock A'] = eq_index[0]
    MARKET_DATA_MANAGER.index_table['stock B'] = eq_index[1]
    MARKET_DATA_MANAGER.scen_gen_table['stock A'] = simulator
    MARKET_DATA_MANAGER.scen_gen_table['stock B'] = simulator

    fund_info = {'Fund A':
                     {
                         'Allocations': {
                             'stock A': 1,
                             'stock B': 0,
                         },
                         'Management Fee': 0.01,
                         'Description': 'blah blah',
                     },
                 'Fund B':
                     {
                         'Allocations': {
                             'stock A': 0,
                             'stock B': 1,
                         },
                         'Management Fee': 0.01,
                         'Description': 'blah blah',
                     },
    }

    credit_rider = isr.InsCreditRateMutualFunds(fund_info=fund_info)

    # Set up non-rider fees
    annual_fee_rate = 0.01
    annual_booking_fee = 100
    mgmt_fee = mif.InsFeeProp(annual_fee_rate, fee_name="Mgmt Fee")
    booking_fee = mif.InsFeeConst(annual_booking_fee, fee_name="Booking Fee")
    fees = [mgmt_fee, booking_fee]

    # Set up rider
    db_rider_fee_rate = 0.005
    db_rider = mir.InsRiderDB(extract_strict(raw_input, "ROP Amount"), db_rider_fee_rate, rider_name="UWL")
    riders = [db_rider]

    # Setup investment index
    inv_index = credit_rider.inv_index(init_date, periods, step_per_year)

    # Setup iteration
    product = InsProduct(riders, fees, inv_index)
    acct = InsAcct(raw_input, product)
    acct_iter = acct.acct_iterator()

    # Setup lapse function and lapse model
    xs = [0]
    ys = [0.0, 0.1]
    shock_func = linear_comp_bounded(1, 0, floor=0.5, cap=1.5)
    lapse_model = LapseDynamic(InsStepFunc(xs, ys), shock_func, rider_name='UWL')

    # Setup surrender charge
    xs = [0]
    ys = [100, 100]
    fixed_charge_func = InsStepFunc(xs, ys)
    xs = [0, 1, 2]
    ys = [0.0, 0.3, 0.2, 0.0]
    pct_charge_func = InsStepFunc(xs, ys)
    surrender_charge = SurrenderCharge(fixed_charge_func, pct_charge_func)

    # Setup mortality function and mortality model
    xs = [x for x in range(0, 100)]
    ys = [0.01] * 100
    ys.append(float('inf'))
    mort_model = InsMortModel(InsStepFunc(xs, ys))

    # Setup VA Model
    model = InsModelVA(acct, lapse_model, mort_model, surrender_charge)
    model_iter = model.create_iterator(pricing_date)

    # model iterator to evolve the model_iter to move forward
    metrics = ['Account Value',
               'Active Population',
               'Benefit Base.UWL',
               'Rider Fee.UWL',
               'Benefit.UWL',
               'Fee.Mgmt Fee',
               'Fee.Booking Fee',
               'Date',
               'Attained Age',
               'Anniv Flag',
               'Death',
               'Lapse',
               'Paid Benefit.UWL',
               'Surrender Charge',
               ]
    crv_aggregator = create_curve_aggregator(metrics)

    params = {'pricing date': init_date, 'periods': 60, 'frequency': 'MS'}
    proj_mgr = ProjectionManager(crv_aggregator, model_iter, **params)
    proj_mgr.run()

    df = crv_aggregator.to_dataframe()
    # df[['Rider Fee.UWL', 'Fee.Mgmt Fee', 'Fee.Booking Fee', 'Surrender Charge']].plot(kind='bar', stacked=True)
    print(df)

    import os
    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    df.to_csv(OUTPUT_PATH + '/va_rider_rs.csv')
    print('output saved to %s' % OUTPUT_PATH + '/rider_rs.csv')

if __name__ == "__main__":
    main()

__author__ = 'Ting'

"""
Utility function to create the shocks of input
"""


def parallel_shift(crv, shift_by):
    crv.data += shift_by


def scale(crv, multiple):
    crv.data = crv.data * multiple


def add_spread(crv, spread_crv):
    """
    This one is trickier
    """
    all_dates = crv.data.index.append(spread_crv.data.index).unique()
    for _date in all_dates:
        if _date not in crv.data.index:
            set_value_at(crv, _date, crv.index_value(_date))
        if _date not in spread_crv.data.index:
            set_value_at(spread_crv, _date, spread_crv.index_value(_date))
    crv.data += spread_crv.data


def shift_knots(crv, shift_by):
    crv.data.index = crv.data.index + shift_by


def set_value_at(crv, date, set_value):
    crv.set_value_at(date, set_value)


# ---------
if __name__ == '__main__':
    import pandas as pd
    import datetime as dt
    import numpy as np
    from Infra.IndexProvider import IndexProvider

    curve = IndexProvider(pd.TimeSeries(index=pd.date_range(start=dt.date(2011, 1, 1), periods=10, freq='D').date,
                                        data=np.arange(10))
                          )
    spread = IndexProvider(pd.TimeSeries(index=pd.date_range(start=dt.date(2011, 1, 1), periods=3, freq='3D').date,
                                         data=np.arange(3))
                           )
    # parallel_shift(curve, 2)
    # print(curve.data)
    # scale(curve, 1.1)
    # print(curve.data)
    print('curve', curve.data)
    print('spread', spread.data)
    add_spread(curve, spread)
    print(curve.data)
    # set_value_at(curve, dt.date(2011, 1, 2), 10)
    # print(curve.data)
    # shift_knots(curve, dt.timedelta(days=1))
    # print(curve.data)
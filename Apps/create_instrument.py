import datetime as dt
from Securities.AssBond import AssBondFixRate
from utils.database import pickle_s

def gen_instrument():
    from lib.utils import dcf_act_act
    # Setup a bond
    parameters = {
        'face': 1344581.6,
        'coupon_rate': 0.03,
        'frequency': 4,
        'issue_date': dt.date(2013, 2, 1),
        'expiration_date': dt.date(2016, 2, 1),
        'dcf': dcf_act_act,
        'pricing_model': None,
        'name': 'Sample Bond 1',
    }

    asset = AssBondFixRate(**parameters)
    pickle_save(asset, 'securities/fixed_rate_bond_sample_1')
    return parameters

if __name__ == "__main__":
    gen_instrument()
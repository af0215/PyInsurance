
"""
create account by combining a product and a policy holder into the account.
"""

from utils.database import pickle_load, pickle_save
from Account.InsAcct import InsAcct


def gen_account():
    policy_holder = pickle_load('policy_holders/ind_000001')
    ins_product = pickle_load('products/product_sample_1')
    acct = InsAcct(policy_holder, ins_product)
    pickle_save(acct, 'accounts/acct_sample_1')


if __name__ == '__main__':
    gen_account()

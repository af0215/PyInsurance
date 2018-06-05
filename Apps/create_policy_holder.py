"""
App to create policy holder information, can set up an web UI for this
"""

import datetime as dt
from utils.database import pickle_save


def create_contract_manual(acct_info):
    if 'ID' not in acct_info:
        raise Exception('ID of this account is not provided!')
    pickle_save(acct_info, 'policy_holders/ind_'+acct_info.get('ID'))


def create_contract_from_excel():
    pass


def generate_policy_holder():
    # Currently put an example here, will be replaced by a real (?UI) based interface for account creation

    raw_input = {"Acct Value": 1344581.6,
                 "Attained Age": 52.8,
                 "ID": "000001",
                 "Issue Age": 45.1,
                 "Issue Date": dt.date(2005, 6, 22),
                 "Initial Date": dt.date(2013, 2, 1),
                 "Maturity Age": 90,
                 "Population": 1,
                 "Riders": dict(),
                 "ROP Amount": 1038872.0,
                 "Gender": "F",
                 "RPB": 1038872.0,
                 "Free Withdrawal Rate": 0.1,
                 "Asset Names": ["Credit Account"],
                 "Asset Values": [1344581.6]}
    create_contract_manual(raw_input)
    return raw_input

if __name__ == '__main__':
    generate_policy_holder()
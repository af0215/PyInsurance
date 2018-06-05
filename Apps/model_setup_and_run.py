"""
Setup the model and run the projection (including diddles)
"""
from utils.database import pickle_load
from utils.projection import run_projection
import datetime as dt
from Models.InsModelFA import InsModelFA
from Models.AssModel import AssModelBase, NoDefaultCreditModel

# setup
step_per_year = 12
periods = 360
init_date = dt.date(2013, 2, 1)
pricing_date = init_date

# TODO: Technically, all the assumptions can be saved or generate on the fly or interact with user input

# ------------ create liability ------------------
acct = pickle_load('accounts/acct_sample_1')

# temporary way to get the assumption objects before I solve the pickle issue
from Apps.create_model_assumptions import *

liab_model = InsModelFA(acct, lapse_model, mort_model, surrender_charge)
liab_model_iter = liab_model.create_iterator(pricing_date)

# ------------ create asset ----------------------
bond = pickle_load('securities/fixed_rate_bond_sample_1')
# Setup a credit model using lapse static
credit_model = NoDefaultCreditModel()

# ------------ Setup Asset Model -------------------
asset_model = AssModelBase(bond, credit_model)
asset_model_iter = asset_model.create_iterator(pricing_date)

# ------------ Setup Liability Models ----------------
liab_metrics = ['CF_Out:Benefit.UWL',
                'CF_In:Rider Fee.UWL',
                'CF_In:Fee.Mgmt Fee',
                'CF_In:Fee.Booking Fee',
                'Date',
                'CF_In:Surrender Charge']
asset_metrics = ['Date',
                 'CF_In:Interest',
                 'CF_In:Principal']

params = {'pricing date': init_date, 'periods': 60, 'frequency': 'MS'}

df = run_projection([liab_model_iter, asset_model_iter], [liab_metrics, asset_metrics], [params, params])

print(df)



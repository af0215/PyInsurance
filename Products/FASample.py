import datetime as dt
import Products.InsCreditRate as isr
import Products.InsFee as mif
import Products.InsRider as mir
from lib.utils import extract_strict
from Products.InsProduct import InsProduct

# Here is an example of creating a product family and then a specified product

# --- Create a new product family ----------
# parameters, TODO: all these redundant info are due to the current way of creating index!
step_per_year = 12
periods = 360
init_date = dt.date(2013, 2, 1)
pricing_date = init_date

# Dummy place holder
TBD = 0.0

# riders
db_rider = mir.InsRiderDB(rider_name="UWL")
riders = [db_rider]

# fees
mgmt_fee = mif.InsFeeProp(fee_name="Mgmt Fee")
booking_fee = mif.InsFeeConst(fee_name="Booking Fee")
fees = [mgmt_fee, booking_fee]

# inv_index, Todo: This part is nasty since the index is 'hard' created for now. Once the logic of searching from
# database is set up, this should be much better.
credit_rider = isr.InsCreditRateFixed()


# define the product family
class FASample(InsProduct):
    def __init__(self, default_params):
        self._default_params = default_params

        self._riders = riders
        self._riders[0].set_fee_rate(default_params.get('DB Rider Fee Rate'))

        self._fees = fees
        self._fees[0].set_fee_rate(default_params.get('Mgmt Fee Rate'))
        self._fees[1].set_fee(default_params.get('Booking Fee'))

        credit_rider.set_credit_rate(default_params.get('Credit Rate'))
        self._inv_index = credit_rider.inv_index(init_date, periods, step_per_year)

    def bind_to_cell(self, cell):
        self._riders[0].update_benefit_base(extract_strict(cell, "ROP Amount"))


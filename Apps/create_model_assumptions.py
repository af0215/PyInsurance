from Models.InsMortModel import InsMortModel
from Models.InsLapseModel import SurrenderCharge, LapseDynamic
from lib.insurance import InsStepFunc, linear_comp_bounded
from utils.database import pickle_save

#TODO: Technically, all the assumptions can be saved or generate on the fly or interact with user input
#TODO: But pickel it is not working now due to the use of decorator I believe.

# Setup lapse function and lapse model
xs = [0]
ys = [0.0, 0.1]
shock_func = linear_comp_bounded(1, 0, floor=0.5, cap=1.5)
lapse_model = LapseDynamic(InsStepFunc(xs, ys), shock_func, rider_name='UWL')
# pickle_save(lapse_model, 'lapse_model_sample_1')

# Setup surrender charge
xs = [0]
ys = [100, 100]
fixed_charge_func = InsStepFunc(xs, ys)
xs = [0, 1, 2]
ys = [0.0, 0.3, 0.2, 0.0]
pct_charge_func = InsStepFunc(xs, ys)
surrender_charge = SurrenderCharge(fixed_charge_func, pct_charge_func)
# pickle_save(surrender_charge, 'surrender_charge_sample_1')

# Setup mortality function and mortality model
xs = [x for x in range(0, 100)]
ys = [0.01] * 100
ys.append(float('inf'))
mort_model = InsMortModel(InsStepFunc(xs, ys))
# pickle_save(mort_model, 'mort_model_sample_1')

DAYS_PER_YEAR = 365
BDAYS_PER_YEAR = 252

FA_MODEL_STATES = ["Active", "Surrendered", "Dead"]
FA_MODEL_STATE_INDICES = {"Active": 0, "Surrendered": 1, "Dead": 2}

BOND_COUPON_FREQUENCY = [1, 2, 4, 12]  # number of payments per year, bad naming i know

BOND_MODEL_STATES = ["Current", "Default"]

EPSILON_WEIGHTS = 1e-5

MODEL_SCRIPTS = {'VA': 'Models.InsModelVA',
                 'FA': 'Models.InsModelFA',
                }
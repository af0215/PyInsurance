''' a class for mortality curve'''
''' should handle survival probability, given various mortality curve setup'''
''' should also handle shocks'''

from SurvivalCurve import SurvivalCurve

class MortalityCurve(SurvivalCurve):
    def __init__(self):
        super(MortalityCurve, self).__init__(None, None)
        """TODO: also want to hard code some canned curves,
        tricky part is the two-dimension part: select and ultimate
        and various interpolation methods"""
        print('this is where i want to implement a morTingtality curve which extends survival curve')
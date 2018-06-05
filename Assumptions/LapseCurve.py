''' a class for lapse curve'''
''' should handle survival probability, given various lapse curve setup'''
''' should also handle shocks'''

from SurvivalCurve import SurvivalCurve

class LapseCurve(SurvivalCurve):
    def __init__(self):
        super(LapseCurve, self).__init__(None, None)
        print('this is where i want to implement a lapse curve which extends survival curve')
Ting
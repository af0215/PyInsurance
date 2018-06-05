''' a generic class for handling survival curve calculation '''
EPSILON = 1e-10

from math import log, exp


class SurvivalCurve():
    def __init__(self, rdates, rvalues):
        """ TODO: i need to define some of the parameters here:
        curve: a relative time curve or a absolute time curve
        anchor: if a relative time curve, needs a start date
        shocks: might need a separate class for handling shocks
        """

        # rdates: relative dates, can be any type that is able to add to datetime
        # and calculate distance between two points.
        assert rdates, "invalid dates"
        assert rvalues, "invalid values"
        assert len(rdates) > 0, "empty dates"
        assert len(rvalues) > 0, "empty values"
        assert len(rdates) == len(rvalues), "dates size %s not compatible with values size %s" % \
                                            (len(rdates), len(rvalues))
        assert all(0 <= v <= 1 for v in rvalues), "attrition rate should be between 0 and 1"

        """TODO: uniqueness concern"""

        self._rdates = rdates
        self._rvalues = [float(x) for x in rvalues]

        zip_curve = zip(rdates, rvalues)

        zip_curve.sort()
        # setting 0 point if missing,
        # with the assumption that rdate * 0 gives 0 point
        dict_curve = dict(zip_curve)
        dict_curve[rdates[0].max] = dict_curve.values()[-1]
        dict_curve[rdates[0]*0] = dict_curve.get(rdates[0]*0, 0)

        curve = dict_curve.items()
        curve.sort()

        self._curve = curve

    def eval(self, rdate):
        """TODO: this is to evaluate the curve at a particular point
        Also, define an operator []"""
        negative_log_curve = [(k, -log(1-v or EPSILON)) for (k, v) in self._curve]
        cumsum = 0

        last_knot = negative_log_curve[0][0]
        last_knot_value = negative_log_curve[0][1]

        for (k, v) in negative_log_curve[1:]:
            dist_to_last_knot = min(k, rdate) - last_knot
            cumsum += (dist_to_last_knot.days / 365.0) * last_knot_value
            last_knot = k
            last_knot_value = v
            if rdate <= k:
                break

        return exp(-cumsum)

    def rate_at(self, rdate):
        """TODO: instantenous rate at a particular date"""
        """TODO: also implement a linear interp in rates space"""
        """I assume the following convention in specifying curve:
            t_0: v1,
            t_1: v2,
            meaning from t_0 to t_1 = v1
            t_1 onwards = v2
        """

        assert self._curve, "Curve is not properly constructed"
        assert rdate > rdate * 0, "Evaluating curve before 0"

        #ast_knot = self._curve[0][0]
        last_knot_value = self._curve[0][1]

        for (k, v) in self._curve[1:]:
            if rdate < k:  # find the left knot and use its value
                return last_knot_value

            #last_knot = k
            last_knot_value = v

        # when exhausted, return right most value. Since i am using a maxi right most knot, shouldnt need this
        return last_knot_value

    def rate_between(self, from_date, to_date):
        """
        this needs to be the effective rate from fromDate to toDate
        :param fromDate:
        :param toDate:
        :return:
        """
        print("override the above")

    def plot(self):
        """
        TODO: need a visualization
        :return:
        """

    def show(self):
        """
        TODO: show several points on the curve
        :return:
        """


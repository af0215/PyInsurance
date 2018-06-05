''' this is the module that consumes assumption files and process the raw input'''


class Assumptions():
    def __init__(self, assumption_filename):
        self.assumption_filename = assumption_filename

    def mortality(self):
        """return a class of mortality curve object"""
        pass

    def lapse(self):
        """return a class of lapse curve object"""
        pass

    # TODO: clearly define what needs to be here

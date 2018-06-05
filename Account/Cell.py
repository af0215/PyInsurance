import datetime as dt
# from lib.decisiontree import look_up


class Cell(object):
    def __init__(self, cell_dict):
        """create a cell from a dictionary"""
        self._raw_input = cell_dict
        self._assumptions = None

    def get_attribute(self, tag):
        if tag in self._raw_input:
            return self._raw_input.get(tag)
        else:
            raise Exception("Attribute \"{}\" does not exist in the cell".format(tag))

    def set_attribute(self, tag, set_value):
        self._raw_input[tag] = set_value

    @property
    def assumptions(self):
        return self._assumptions

    @assumptions.setter
    def assumptions(self, assumption_dict):
        pass  # TODO: implement the method to set the assumptions like "cell.assumptions = assumption_dict"

    def get(self):
        pass  # TODO: implement so that it behaves the same as the dict.get()

#-- Example --
if __name__ == "__main__":
    raw_input = {
        "Acct Value": 1344581.6,
        "Attained Age": 52.8,
        "DB Rider Name": "Step-up",
        "WB Rider Name": "PP",
        "ID": "000001",
        "Issue Age": 45.1,
        "Issue Date": dt.date(2005, 6, 22),
        "Maturity Age": 90,
        "Population": 1,
        "Riders": dict({}),
        "ROP Amount": 1038872.0,
        "Gender": "F",
        "RPB": 1038872.0,
        "Free Withdrawal Rate": 0.1,
        "Sub-accounts": ["Conservative", "Moderate"],
        "Sub-account Quantities": [1344581.6, 0]
    }
    sample_cell = Cell(raw_input)
    print(sample_cell.get_attribute("Acct Value"))
    sample_cell.set_attribute("Initial Date", dt.date(2013, 3, 28))
    print(sample_cell.get_attribute("Initial Date"))
    # print(sample_cell.get_attribute("Withdrawal"))

"""
A class to put together fees, riders, investment index and all the information (logic) that is not dependent
on the policy holder. The purpose of this is to
1. make the structure more cleaner, conceptually, Account = policy holder + product
2. make the development of product cleaner and easier as
    Generic Product (class, like InsFA) -> product family (class, with fixed logic but undetermined parameters,
    like Ping An's specific product) -> specific product (object, with parameters fixed)
   This setup, I think, will facilitate fast development and make the trial and error for product design
"""


class InsProduct(object):
    """
        This is a generic insurance product, one can create more specific class by specifying riders,
        fees, and, and inv_index logic to create a product family, and create member product object
        by setting default_parameters in each product family
    """
    def __init__(self,
                 riders,
                 fees,
                 inv_index,
                 default_params=None):
        self._riders = riders
        self._fees = fees
        self._inv_index = inv_index
        self._default_params = default_params

    def bind_to_cell(self, cell):
        pass

    @property
    def riders(self):
        return self._riders

    @property
    def fees(self):
        return self._fees

    @property
    def inv_index(self):
        return self._inv_index


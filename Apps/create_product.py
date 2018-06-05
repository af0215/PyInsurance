from Products.FASample import FASample
from utils.database import pickle_save


def gen_product():
    # create a product object with parameters
    product_defaults = {'Credit Rate': 0.03,
                        'Mgmt Fee Rate': 0.01,
                        'Booking Fee': 100,
                        'DB Rider Fee Rate': 0.005,
                        }

    new_product = FASample(product_defaults)

    pickle_save(new_product, 'products/product_sample_1')
    return product_defaults

if __name__ == "__main__":
    gen_product()


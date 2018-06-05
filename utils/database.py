__author__ = 'Ting'

import pickle
import os

# one level up relative to this particular file, instead of calling file
PROJECT_ROOT = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
PICKLE_DB_PATH = PROJECT_ROOT + '/pickle_db/'


def pickle_save(pickle_obj, pickle_obj_name, db_path=PICKLE_DB_PATH):
    if not pickle_obj_name.endswith('.p'):
        pickle_obj_name += '.p'
        pickle.dump(pickle_obj, open(db_path+pickle_obj_name, 'wb+'))


def pickle_load(pickle_obj_name, db_path=PICKLE_DB_PATH):
    if not pickle_obj_name.endswith('.p'):
        pickle_obj_name += '.p'
    return pickle.load(open(db_path+pickle_obj_name, 'rb'))


def read_from_excel(file_name, pickle_obj_name, db_path=PICKLE_DB_PATH):
    pass


def print_path():
    print(PICKLE_DB_PATH)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    dict_obj = {'a': 1, 'b': 2}
    obj_name = 'test_obj'
    pickle_save(dict_obj, obj_name)
    rec_obj = pickle_load(obj_name)
    print(rec_obj.get('a'))

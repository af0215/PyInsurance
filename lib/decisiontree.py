"""
This is the library for decision tree functions.

The decision tree is a dictionary of the following format:
t = {
    '__key__': 'Gender',
    "F": 65,
    "M": 70,
    "J": {
        '__key__': 'Type',
        "A": 71,
        "B": 72
        }
}
or value for all cases as
t = 20
"""

MAGIC_KEY = '__key__'


def look_up(values, keys):
    if values is None:
        raise Exception("The values is None!")
    if not isinstance(values, dict):
        return values
        #TODO: this one is error-prone
    if MAGIC_KEY not in values:
        raise Exception("\'__key__\' is not in values")
    else:
        return look_up(values.get(keys.get(values.get(MAGIC_KEY))), keys)

t = {
    '__key__': 'Gender',
    "F": 65,
    "M": 70,
    "J": {
        '__key__': 'Type',
        "A": 71,
        "B": 72
    }
}

# -- Example --
if __name__ == "__main__":
    t1 = {
        '__key__': 'Gender',
        "F": 65,
        "M": 70,
        "J": {
            '__key__': 'Type',
            "A": 71,
            "B": 72}}
    t2 = 20

    # use dict as an example, will replace with cell class later
    cell = {
        'Gender': 'M',
        'Type': 'A'
    }
    print(look_up(t1, cell))
    print(look_up(t2, cell))






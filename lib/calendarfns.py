import numpy as np

BUSCAL_CHN = np.busdaycalendar(holidays=['2015-01-01', '2015-02-19'])


def day_count(begin_dates, end_dates, freq='D', calendar=None):
    """
        Currently, it is just a wrapper on datetime and numpy functions.
        Setup the interface here to facilitate future update
        calendar:
        A busdaycalendar object which specifies the valid days.
        If this parameter is provided, neither weekmask nor holidays may be provided.
    """
    if freq == 'D':  # using timedelta to deal with this
        return (end_dates-begin_dates).days
    if freq == 'B':
        if calendar:
            return np.busday_count(begin_dates, end_dates, busdaycal=calendar)
        else:
            return np.busday_count(begin_dates, end_dates)
    else:
        raise NotImplementedError('Other freq type is NOT supported!')

if __name__ == '__main__':
    import datetime as dt
    print(day_count(dt.date(2014, 1, 1), dt.date(2014, 12, 1)))
    print(day_count(dt.date(2014, 1, 1), dt.date(2014, 12, 1), freq='B'))


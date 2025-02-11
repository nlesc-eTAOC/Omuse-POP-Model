import os
import numpy as np
from pathlib import Path
from calendar import monthrange

# Set month range manual, used when not considering leap years
month_range = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def year_month_day(a_date: str) -> tuple[int, int, int]:
    """Return a tuple with year, month and day as ints.

    Args:
        str: a POP formatted date string

    Returns:
        Tuple with [YYYY, MM, DD]
    """
    year = a_date[0:4].lstrip("0")
    month = a_date[4:6].lstrip("0")
    day = a_date[6:8].lstrip("0")
    return int(year), int(month), int(day)


def remainingdaysincurrentmonth(a_date: str,
                                allow_leap : bool = False) -> int:
    """Return the number of days remaining in the month.

    Args:
        str: a POP formatted date string
        bool: accounting for leap year flag

    Returns:
        Number of days remaining in the month
    """
    year, month, day = year_month_day(a_date)
    if allow_leap:
        return monthrange(year, month)[1] - day + 1
    else:
        return month_range[month-1] - day + 1


def daysincurrentyear(a_date: str,
                      allow_leap : bool = False) -> int:
    """Return the number of days in a given year.

    Args:
        str: a POP formatted date string
        bool: accounting for leap year flag

    Returns:
        Number of days in current year
    """
    year, _, _ = year_month_day(a_date)
    if allow_leap:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 366
        else:
            return 365
    else:
        return 365


def monthly_reference(a_date: str) -> float:
    """Return the score function reference for a given date."""
    # Remove one because this is called at the beginning of
    # the month following that of interest
    # Remove another 1 for python zero-indexing
    _, month, _ = year_month_day(a_date)
    month = month - 1 - 1
    if month < 0:
        month = 11
    refs = [
        18.898761578011541,  # 00 Jan
        17.843680173064680,  # 01 Fev
        16.836239483624375,  # 02 Mar
        16.168102755708293,  # 03 Apr
        14.866069622206135,  # 04 May
        14.274935800469697,  # 05 Jun
        15.311850750952376,  # 06 Jul
        15.382891756041191,  # 07 Aug
        14.437490042573950,  # 08 Sep
        15.426391231293266,  # 09 Oct
        18.143483326175577,  # 10 Nov
        18.805042028508758,  # 11 Dec
    ]
    return refs[month]

def random_file_in_list(list_file: str) -> str:
    """Return a entry from a list at random.

    Args:
        list_file : a file containing a list of init files

    Returns:
        a string with the path to an init file
    """
    assert os.path.exists(list_file) is True

    # Load list of init files
    with open(list_file, "r") as lsf:
        list_f = lsf.readlines()

    # Select a random file
    selected = np.random.default_rng().integers(0, len(list_f))
    elected = list_f[selected].strip()

    assert Path(elected).exists() is True

    return elected

from calendar import monthrange


def remainingdaysincurrentmonth(a_date: str) -> int:
    """Return the number of days remaining in the month."""
    year = a_date[0:4].lstrip("0")
    month = a_date[4:6].lstrip("0")
    day = a_date[6:8].lstrip("0")
    remain = monthrange(int(year), int(month))[1] - int(day) + 1
    return remain


def daysincurrentyear(a_date: str, allow_leap : bool = False) -> int:
    """Return the number of days in a given year."""
    year = int(a_date[0:4].lstrip("0"))
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
    month = int(a_date[4:6].lstrip("0")) - 1 - 1
    if month < 0:
        month = 12
    refs = [
        0.00361611655328,  # 00 Jan
        0.0361834048733,  # 01 Fev
        0.0682092340795,  # 02 Mar
        0.133053149707,  # 03 Apr
        0.224648033922,  # 04 May
        0.247606740195,  # 05 Jun
        0.202784065642,  # 06 Jul
        0.192188404163,  # 07 Aug
        0.2046571082,  # 08 Sep
        0.14090089013,  # 09 Oct
        0.0251926198575,  # 10 Nov
        0.00503752090331,  # 11 Dec
        0.00361611655328,
    ]  # 12 Jan
    return refs[month]


# if __name__ == "__main__":
#    date = "20120515"
#    print(remainingdaysincurrentmonth(date))
#    print(monthly_reference(date))

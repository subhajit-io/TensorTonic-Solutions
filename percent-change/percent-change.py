def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    le = len(series)
    result = []
    for i in range(1,le):
        if series[i-1] == 0:
            result.append(0.0)
        else:
            result.append((series[i]-series[i-1])/series[i-1])
    return result
from multipledispatch import dispatch


@dispatch(str)
def range_ord(stop):
    return [chr(ordval) for ordval in range(ord("a"), ord(stop) + 1)]


@dispatch(str, int)
def range_ord(stop, step):
    return [chr(ordval) for ordval in range(ord("a"), ord(stop) + 1, step)]


@dispatch(str, str)
def range_ord(start, stop):
    return [chr(ordval) for ordval in range(ord(start), ord(stop) + 1)]


@dispatch(str, str, int)
def range_ord(start, stop, step):
    return [chr(ordval) for ordval in range(ord(start), ord(stop) + 1, step)]
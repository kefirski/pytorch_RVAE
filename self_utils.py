def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def fand(x, y):
    return x and y


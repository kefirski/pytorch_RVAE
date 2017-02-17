def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parmeters = module.parameters()
    return fold(f_and, parmeters, True) or not fold(f_or, parmeters, False)

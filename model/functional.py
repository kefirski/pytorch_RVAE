def fold(f, l, a):
    return a if (len(l) == 0) else fold(f, l[1:], f(a, l[0]))


def f_and(x, y):
    return x and y


def f_or(x, y):
    return x or y


def parameters_allocation_check(module):
    parmeters = list(module.parameters())
    return fold(f_and, parmeters, True) or not fold(f_or, parmeters, False)


def handle_inputs(inputs, use_cuda):
    import torch as t
    from torch.autograd import Variable

    result = [Variable(t.from_numpy(var)) for var in inputs]
    result = [var.cuda() if use_cuda else var for var in result]

    return result

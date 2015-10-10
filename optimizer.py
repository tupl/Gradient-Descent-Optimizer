import numpy as np
import matplotlib as plt
import random
import math


GOLDEN_RATIO = (math.sqrt(5)-1)/2


def norm(x):
    return np.sqrt(x.dot(x))


def norm2(x):
    return x.dot(x)


def start_point(low=-100, high=100, dim=2):
    init = []
    for i in range(dim):
        init.append(random.randint(low, high))
    return np.array(init, float)


def line_search(func, init_pt, direct, step=0.01, x_abs_err=0.01,
                x_rel_err=0.01, f_abs_err=0.01, f_rel_err=0.01):
    global GOLDEN_RATIO

    x1 = init_pt
    x2 = x1 + step * direct

    if func(x2) > func(x1):
        step = - step

    while True:
        step = step / GOLDEN_RATIO
        x4 = x2 + step * direct

        if func(x4) > func(x2):
            break
        else:
            x1 = x2
            x2 = x4

    old_f = float("inf")

    while True:
        x3 = GOLDEN_RATIO * x4 + (1 - GOLDEN_RATIO) * x1
        if func(x2) < func(x3):

            if norm(x1 - x3) < x_abs_err + x_rel_err * norm(x2):
                return (x1 + x3) / 2

            f1 = func(x1)
            f2 = func(x2)
            f3 = func(x3)
            _f = (f1 + f2 + f3) / 3

            if abs(old_f - _f) < f_abs_err + f_rel_err * abs(_f):
                return (x1 + x3) / 2

            old_f = _f

            x4 = x1
            x1 = x3
        else:
            if norm(x2 - x4) < x_abs_err + x_rel_err * norm(x3):
                return (x2 + x4) / 2

            f2 = func(x2)
            f3 = func(x3)
            f4 = func(x4)
            _f = (f2 + f3 + f4) / 3

            if abs(old_f - _f) < f_abs_err + f_rel_err * abs(_f):
                return (x2 + x4) / 2

            old_f = _f

            x1 = x2
            x2 = x3


def steepest_descent(func, grad_func, init_point, num_iter=100,
                     dim=2, x_err=0.001, g_err=0.001, line_search=line_search):
    x = init_point

    for i in range(num_iter):
        grad = - grad_func(x)
        new_x = line_search(func, x, grad)

        if norm(grad_func(new_x)) < g_err or norm(new_x - x) < x_err:
            return new_x
        x = new_x
    return x


def cyclic_coordinate(func, init_point, dim=2, num_iter=100,
                      line_search=line_search, x_err=0.001, f_err=0.001):
    x = init_point

    for i in range(num_iter):

        old_x = np.copy(x)

        accel_list = []

        for j in range(dim):
            zero = [0 for k in range(dim)]
            direct = np.array(zero, float)
            direct[j] = 1.0
            new_x = line_search(func, x, direct)

            accel_list.append(new_x - x)

            x = new_x

        accel = sum(accel_list)
        x = line_search(func, x, accel)

        if norm(x - old_x) < x_err:
            return x
        if abs(func(x) - func(old_x)) < f_err:
            return x
    return x


def conjugate_gradient(func, grad_func, init_point, num_iter=100,
              dim=2, g_err=0.001, line_search=line_search):

    x = [0 for i in range(dim + 1)]
    alpha = [0 for i in range(dim + 1)]
    d = [0 for i in range(dim + 1)]
    b = [0 for i in range(dim + 1)]

    x[0] = init_point

    for i in range(num_iter):

        d[0] = - grad_func(x[0])

        for k in range(dim):

            new_x = line_search(func, x[k], d[k])

            alpha[k] = (new_x - x[k])[0] / d[k][0]

            x[k + 1] = x[k] + alpha[k] * d[k]

            b[k] = norm2(grad_func(x[k + 1])) / norm2(grad_func(x[k]))

            d[k + 1] = - grad_func(x[k + 1]) + b[k] * d[k]

            if norm(grad_func(x[k+1])) < g_err:
                return x[k+1]

        x[0] = x[dim]
    return x[0]

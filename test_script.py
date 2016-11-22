# -*- coding: utf-8 -*-
"""
Author : JasonGUTU
Email  : hellojasongt@gmail.com
Python : anaconda3
Date   : 2016/11/18

test"""
import math

import numpy as np

from naa.naa import NAA

def GeneralizedGriewank(variables):
    assert isinstance(variables, np.ndarray), "Argumrnts by a np.array obj with type float."
    prod = math.cos(variables[0])
    for i in range(1, len(variables)):
        prod *= math.cos(variables[i] / math.sqrt(i + 1))
    sqr = variables ** 2
    sum_ = sqr.sum() / 4000
    return sum_ - prod + 1

bound_p = np.array([[-5., -5., -5., -5., -5., -5., -5., -5., -5., -5.], [ 5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.]])
param = [100, 5, 15, 0.5, 0.5, 1.0, 1.0]

solver = NAA(10, bound_p, 100, param)
solver.load_problem(GeneralizedGriewank)

solver.solve(prt=True)


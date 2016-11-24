# -*- coding: utf-8 -*-
"""
Author : JasonGUTU
Email  : hellojasongt@gmail.com
Python : anaconda3
Date   : 2016/11/22

Several testing function.
All the problems has same properties:
    Argumrnts by a np.array obj with type float
    Return is float"""

import math

import numpy as np


def SphereModel(variables):
    assert isinstance(variables, np.ndarray), "Argumrnts by a np.array obj with type float."
    sqr = variables ** 2
    return sqr.sum()

def GeneralizedGriewank(variables):
    assert isinstance(variables, np.ndarray), "Argumrnts by a np.array obj with type float."
    prod = math.cos(variables[0])
    for i in range(1, len(variables)):
        prod *= math.cos(variables[i] / math.sqrt(i + 1))
    sqr = variables ** 2
    sum_ = sqr.sum() / 4000
    return sum_ - prod + 1

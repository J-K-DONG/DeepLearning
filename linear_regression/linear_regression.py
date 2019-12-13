#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3
@author: JK_DONG
@software: PyCharm
@file: linear_regression.py
@time: 2019-12-13 21:29

"""

import numpy as np


def compute_lost_for_line_points(points, b, w):
    lost = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        lost += (y - (w * x + b)) ** 2
    lost_sum = lost / float(len(points))
    return lost_sum


def step_gradient(points, b_current, w_current, learning_rate):

    b_step = 0
    w_step = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        b_step += ((w_current * x + b_current) - y)
        w_step += ((w_current * x + b_current) - y) * x
    b_new = b_current - (b_step * learning_rate * 2 / float(len(points)))
    w_new = w_current - (w_step * learning_rate * 2 / float(len(points)))
    return [b_new, w_new]


def gradient_decent_runnign(points, b_current, w_current, learning_rate, num_iterations):

    for i in range(num_iterations):
        b_current, w_current = step_gradient(points, b_current, w_current, learning_rate)
    return b_current, w_current


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    b_start = 0
    w_start = 0
    num_iterations = 1000
    learning_rate = 0.0001
    print("the lost of beginning is : {}".format(compute_lost_for_line_points(points, b_start, w_start)))
    [b_end, w_end] = gradient_decent_runnign(points, b_start, w_start, learning_rate, num_iterations)

    print("after {0} gradients the current b is : {1}, and the current w is : {2},  the lost of points is : {3}"
          .format(num_iterations, b_end, w_end, compute_lost_for_line_points(points, b_end, w_end)))


if __name__ == "__main__":
    run()
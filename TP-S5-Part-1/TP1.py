#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:13:06 2026

@author: elhadj
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

U = np.array(
    [10.24, 10.79, 11.34, 12.29, 13.33, 14.43, 15.22, 16.24, 17.22, 18.23]
)
I = np.array([5.29, 5.45, 5.60, 5.86, 6.12, 6.40, 6.61, 6.82, 7.05, 7.27])

T_0 = (
    np.array(
        [
            1724.9,
            1761.6,
            1794.8,
            1848.8,
            1906.1,
            1958.9,
            2001.7,
            2042.3,
            2086.4,
            2132.6,
        ]
    )
    + 273.15
)

R = U / I

R_amb = 0.169

rho_amb = 5.65e-8

alpha = R_amb / rho_amb


rho_T = R / alpha

T = (rho_T + 7.1e-8) / (0.0320e-8)

h = 6.626e-34
c = 2.99e8
lbd = 0.85e-6
k = 1.38e-23
c_1 = 8 * np.pi * h * c / (lbd**5)
c_2 = h * c / (k)

e_lbd_T = c_1 * np.exp(-c_2 / (lbd * T))
e_lbd_T_0 = c_1 * np.exp(-c_2 / (lbd * T_0))
epsilon_lbd = e_lbd_T_0 / e_lbd_T

epsilon_moy = np.mean(epsilon_lbd)


X = np.exp(-c_2 / (lbd * T))

Y = np.exp(-c_2 / (lbd * T_0))

plt.plot(X, Y, "+")


L = linregress(X, Y)
A = L[0]
B = L[1]
C = L[-2]
plt.plot(X, A * X + B)
plt.show()

P = R * I * I
Ln_P = np.log(P)
Ln_T = np.log(T)
plt.plot(Ln_T, Ln_P, "+")
L2 = linregress(Ln_T, Ln_P)
s = L2[0]
B2 = L2[1]
C2 = L2[-2]
plt.plot(Ln_T, s * Ln_T + L2[1])
plt.xlabel("Log(T)")
plt.ylabel("f(Log(T))")
plt.show()


r_s = P ** (1 / 4)
plt.plot(r_s, R, "+")

L3 = linregress(r_s, R)
alpha_a = L3[0]
B3 = L3[1]
Y3 = alpha_a * 0.0320e-8 * r_s + B3
plt.plot(r_s, Y3)
plt.xlabel("Racine s ième de  P")
plt.ylabel("R")
plt.show()

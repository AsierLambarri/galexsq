import numpy as np
from math import log
from unyt import unyt_quantity, unyt_array
from numba import (
    vectorize,
    float32,
    float64,
    njit,
    jit,
    prange,
    get_num_threads,
    typed
)
from scipy.integrate import quad
from copy import copy

def NFWc(r, rho_h, c, Rvir):
    x = r / Rvir
    A_nfw = np.log(1 + c) - c / (1 + c)
    return rho_h / (3 * A_nfw * x * (1/c + x)**2)

def TruncatedNFWc(r, rho_h, c, Rvir, rt, eta = 1):
    return NFWc(r, rho_h, c, Rvir) * ( rt**2 / (rt**2 + r**2) )**eta



def NFWrs(r, rho_0, Rs):
    x = r / Rs
    return rho_0 / ( x * (1 + x)**2 )

def TruncatedNFWrs(r, rho_0, Rs, rt, eta = 1):
    return NFWrs(r, rho_0, Rs) * ( rt**2 / (rt**2 + r**2) )**eta



def Einasto(r, rho_s, alpha, Rs):
    x = r / Rs
    return rho_s * np.exp( -2/alpha * ( (x)**alpha - 1) )



def mEnc_NFWc(r, rho_h, c, Rvir):
    
    return np.array([quad(lambda u: 4 * np.pi * u**2 * NFWc(u, rho_h, c, Rvir), 0, ri)[0] for ri in r]) 

def vcirc_NFWc(r, rho_h, c, Rvir):
    return np.sqrt( mEnc_NFWc(r, rho_h, c, Rvir) / r * 4.300917270038e-06)
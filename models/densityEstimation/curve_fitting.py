
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import splrep, BSpline, make_smoothing_spline
import datetime as dt 


def bspline(x, y, k = 3):
    """ 
    Return a B-Spline for the given x and y values 
    Args:
        x: x values
        y: y values
        k: degree of the spline
    
    Returns:
        tuple: x values, y values
    
    """
    if x.shape[0] < 4:
        return x, y
    tck = splrep(x, y, k = k, s = len(x))
    tx = splrep(x, y, s = 0 )
    return tx[0], BSpline(*tck)(tx[0])


def smoothing_spline(x, y):
    """ 
    Return a smoothing spline for the given x and y values 
    
    Args:
        x: x values
        y: y values
    
    Returns:
        tuple: x values, y values
    
    """
    if x.shape[0] < 4:
        return x, y
    tck = make_smoothing_spline(x, y)
    return x, tck(x)
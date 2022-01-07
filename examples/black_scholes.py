# -*- coding: utf-8 -*-
import sys
sys.path.append(r"D:\clouds\github\quantfinance-examples-dev")
sys.path.append(r"D:\clouds\github\quantfinance-examples-dev\pyqfin")
from pyqfin.models.black_scholes import Parameters, Analytic

params = Parameters(sigma=0.2, r=0.03)
a = Analytic(params)


'''
Created on Jul 30, 2020
@author: fritz
'''
import numpy as np
import plots

plots.plot_complex_eigvals(show_plots=True,N = 4, k = 1, T = 1, dt = 1e-5)
#plots.plot_test(show_plots=True,N = 4, k = 1, T = 20, dt = 1e-3)
#plots.plot_basic_model(show_plots=True, N=4, k = 1, T = 50, dt = 1e-3)
# plots.plot_basic_model_gj(show_plots=True, N=4, k = 1, T = 40, dt = 1e-3)
# plots.plot_const_driving(show_plots=True,T=10, dt=1e-4)
# plots.plot_pcf_gj_sc_comparison(show_plots=True, N=8, T=10, dt=1e-4)


plots.plot_sho(['pcf', 'gj'], N = 32, T = 50, dt = 1e-3)


# print('\n -----a--------Plotting Complete-------------')

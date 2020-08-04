'''
Created on Jul 30, 2020
@author: fritz
'''

import plots 

#plots.plot_basic_model(show_plots = True)

#plots.plot_const_driving(show_plots=False, dt=1e-4)

#plots.plot_const_dynamical_system(show_plots=False, dt=1e-4)

plots.plot_pcf_gj_sc_comparison(show_plots=True, N=32, T=10, dt=1e-4)

print('\n -------------Plotting Complete-------------') 
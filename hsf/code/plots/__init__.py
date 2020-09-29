'''
Plots for HSF Research
'''


# Add Base Directory for HSFNets and utils packages here.

path_to_Misc_Research_Code = '/home/chris/Desktop/git_repos/Misc-Research-Code'

 
print(path_to_Misc_Research_Code)
import sys
sys.path.append(path_to_Misc_Research_Code)
import os



# Import Network Simulation Tools
import_success = True
try:
    from HSF.HSFNets import *
except Exception as e: 
    print('Could Not Import HSFNets - {0}'.format(e))
    import_success = False
try:
    from utils import *
except Exception as e:
    print('Could Not Import utils - {0}'.format(e))
    import_success = False

if not import_success:
    exit()
else:
    print('Package Imports Successful\n')


# Configure Plot Default Parameters
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.markersize'] = 10
plt.rcParams['lines.linewidth'] = 4

# Functions for Self-Coupled Network
def rmse_sp_ksl(k,s,l):
    t1 = k**2
    

    
    p = phi(s/k, l)
    
    
    t2 = p*s*k
    
    t3 = p * s**2 / 2 * ((1 + np.exp(-1/p)))/((1 - np.exp(-1/p)))
    
#     plt.figure()
#     plt.plot(t1,label='t1')
#     plt.plot(-2*t2,label='-2t2')
#     plt.plot(t3,label='t3')
#     plt.plot(t1-2*t2,label='t1-2*t2')
#     plt.plot(t1-2*t2+t3,label='t1-2*t2+t3')
#     plt.legend()
#     plt.show()
#     print(np.sqrt(t1 +2*t2 + t3 ))
#     assert(False)

    return np.sqrt(t1 -2*t2 + t3 )
    
def rmse_sp_k(k):
    p = phi_k(k)
    top = 1 - np.exp( - 2 / p)
    bot = 2 * (1 - np.exp(-1 / p) )**2
    return np.sqrt(
        k**2 + p * (top / bot - 2 * k)
        )

def phi(sk, lam=1):
    t1 = np.log( 1 + lam * sk / 2)

    t2 = np.log( 1 - lam * sk / 2)
    return lam * (t1 - t2)**-1

def rmse_t(xhat, xtru, ts):
    ndims = xhat.shape[0]
    rmse = 0
    delta_t = ts[1]  - ts[0]
    start = len(ts) // 2 
    for i in range(ndims):
        e = xtru[i,start:] - xhat[i,start:]
        se = np.square(e)
        mse =  delta_t * np.sum(se)  / (ts[-1] - ts[start])
        rmse += np.sqrt(mse)
    return rmse

def rmse_phi_const_driving(phi):

    vth = .5
    a = np.exp( - 1 / phi)

    #term 1  
    top = vth + a / 2
    bot = 1 - a
    t1 = (top / bot)**2

    #term 2
    top = 1 - a**2
    bot = 2 * ( 1 - a)**2
    t2 = top / bot

    #term 3
    top = vth + a / 2
    bot = 1 - a
    t3 = top / bot

    return np.sqrt( t1 + phi * ( t2 - 2 * t3) )

def rmse_phi_const_dynamics(phi):
    t1 =  np.tanh(
            1 / (2 * phi )
         )
    return np.sqrt(1 - 2 * phi * t1)

def rmse_phi_s_const_dynamics(s):
    p = phi_s(s)
    return np.sqrt( 1 - p / s)
 
 
# Functions for PCF Network

def pcf_phi(unit_norm_d, ks):  
    c = np.asarray([1, 0])
    if len(list_check(ks)) == 1:
        d = unit_norm_d * ks
        dtc = d.T @ c 
        norm_d = np.linalg.norm(d) ** 2
        logarg1 = dtc + (norm_d / 2)
        logarg2 = dtc - (norm_d / 2)
        return (  
                np.log(logarg1)
                -
                np.log(logarg2)
                )**-1
    else:
        phis = np.zeros((len(ks),))
        for i, k in enumerate(ks):
            d = unit_norm_d * k
            dtc = d.T @ c  
            norm_d = np.linalg.norm(d)**2
            logarg1 = dtc + (norm_d / 2)
            logarg2 = dtc - (norm_d / 2)
            
            phis[i] = (  
                    np.log(logarg1)
                    -
                    np.log(logarg2)
                    )**-1
        return phis
                       
def per_spike_rmse_numerical(data, idx):
    ''' given data from a network sim, compute the average per-spike rmse of neuron indexed by idx'''
    num_spikes = data['spike_nums'][idx]
    spikes = data['O'][idx,:num_spikes]
    
    if num_spikes < 6 :
        print('only {0} spikes occurred'.format(num_spikes))
        return np.nan, np.nan
    
    rmses = []
    
    start_spike = 5
    for trial in np.arange(start_spike, num_spikes - 1):
        left_idx = np.argwhere(spikes[trial] < data['t'][:])[0][0]
        right_idx = np.argwhere(spikes[trial + 1] <  data['t'][:])[0][0]
    
        delta_t = data['t'][1] - data['t'][0]
    

        e = data['x_hat'][0,left_idx:right_idx] - data['x_true'][0,left_idx:right_idx]
    
        se = np.square(e)
        mse =  delta_t * np.sum(se)  / (data['t'][right_idx]-data['t'][left_idx])
        rmses.append(np.sqrt(mse))
    return np.mean(np.asarray(rmses)), np.std(np.asarray(rmses))

def x_hat_explicit_s(t, s): 
            eq_pred_height = (
                s**-1  / (1 - np.exp(- 1 / phi_s(s)))
            )
            return eq_pred_height * np.exp(- np.mod(t, phi_s(s)**-1))

def run_sim(N, p=1, T = 20,  k = 1, dt = 1e-5, stim='const', D_scale = 1, D_type='none'):

    A =  - np.eye(2)
    B = np.eye(2)
    x0 = np.asarray([.5, 0])
    _, uA = np.linalg.eig(A)
    d = A.shape[0]
    mode = '2d cosine'
    if D_type=='none': 
        D = D_scale * np.eye(N)[0:A.shape[0],:]
    elif D_type=='2d cosine':
        D = D_scale * gen_decoder(A.shape[0], N, mode=mode)
    elif D_type=='simple':
        D = D_scale * uA @ np.hstack((
            np.eye(d),
            np.zeros((d, N - d))
            ))
    else:
        D = gen_decoder(A.shape[0], N)
    print('Using {} decoder type.\n'.format(D_type))
    
    
    if stim == 'const':
        sin_func = lambda t :  k * uA[:,0]
    else:
        sin_func = lambda t :  k * np.asarray([np.cos( (1/4) * np.pi*t), np.sin( (1/4) * np.pi*t)])
    
    
    lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
    net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = p)
    data = net.run_sim() 
    

    return data

def check_dir(name):
    '''
    Given a string [name] check if the subdirectory exists in the 
    directory **containing this file**
    if it does, return, if it does not, create it as a subdirectory
    NOTE: written on windows machine so directories contain "/" slashes
    Returns the path of the desired directory whether or not one was 
    created
    '''
    
        
    this_dir = os.path.dirname(os.path.realpath(__file__)) 
    if not os.path.exists(this_dir + '/' + name):
        os.makedirs(this_dir + '/' + name)

    return this_dir + '/' + name



## Plotting Functions

def plot_basic_model(show_plots=True,N = 4, k = 1, T = 10, dt = 1e-5):
    
    name = 'basic_plots'
    this_dir =  check_dir(name)
    
    data = run_sim(N, 1, k = k, T =  T, dt = dt, D_type='simple', stim='sinusoid', D_scale=.1)
    
    plot_step = 10
    
    vmin = -np.min(data['vth'])
    vmax = np.max(data['vth'])
    
    plt.figure()
    for i in range(4):
        plt.plot(data['t'],data['V'][i,:], label='Neuron %i Voltage'%i)
    plt.legend()
    plt.title("Neuron Membrane Potentials")
    plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
    plt.ylabel('Membrane Potential')
    plt.savefig('plots/basic_plots/membrane_potential_plot.png',bbox_inches='tight')
    
    plt.figure()
    cbar_ticks = np.linspace( start = vmin, stop = vmax,  num = 8, endpoint = True)
    plt.imshow(data['V'],extent=[0,data['t'][-1], 0,3],vmax=vmax, vmin=vmin)
    plt.xlabel(r"Dimensionless Units of $\tau$")
    plt.axis('auto')
    cbar = plt.colorbar(ticks=cbar_ticks)
    cbar.set_label(r'$\frac{v_j}{v_{th}}$')
    cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
    plt.title('Neuron Membrane Potentials')
    plt.ylabel('Neuron #')
    plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])
    plt.savefig(this_dir + '/' + 'membrane_potential_image.png',bbox_inches='tight')
    
    plt.figure()
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step],c='r',label='Decoded Network Estimate (Dimension 0)' )
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step],c='g',label='Decoded Network Estimate (Dimension 1)' )
    plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][0,0:-1:plot_step],c='k')
    plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][1,0:-1:plot_step],c='k',label='True Dynamical System')
    plt.title('Network Decode')
    plt.legend()
    plt.ylim([-1.1, 1.1])
    plt.xlabel(r'Dimensionless Time $\tau_s$')
    plt.ylabel('Decoded State')
    plt.savefig(this_dir + '/' + 'network_decode.png',bbox_inches='tight')
    
    plt.figure()
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step] - data['x_true'][0,0:-1:plot_step],c='r',label='Estimation Error (Dimension 0)' )
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step] - data['x_true'][1,0:-1:plot_step],c='g',label='Estimation Error (Dimension 1)' )
    plt.title('Decode Error')
    plt.legend()
    plt.ylim([-1.1, 1.1])
    plt.xlabel(r'Dimensionless Time $\tau_s$')
    plt.ylabel('Decode Error')
    plt.savefig(this_dir + '/' + 'decode_error.png',bbox_inches='tight')

#     plt.figure()
#     for i in range(4):
#         plt.plot(data['t'][0:-1:plot_step], data['r'][i,0:-1:plot_step])
    
    

    if show_plots:
        plt.show()

def plot_const_driving(show_plots=True,N = 4, T = 10, dt = 1e-5):
      
    def run_plots():
        #plot_voltage_prediction() 
        #plot_rate_vs_k()
        #plot_xhat_estimate_explicit()
        plot_per_spike_rmse_vs_k_phi() 
        
        if show_plots:
            plt.show()

    def plot_voltage_prediction():
        def v(s, k, v0, r0, t, lam):
            t1a = np.exp(lam * t)
            t1b = (s.T @ k / lam + s.T @ s * r0 + v0)
            t2 = s.T @ s * r0 * np.exp(-t)
            t3 = s.T @ k / lam
            return np.squeeze(t1a * t1b - t2 - t3)
            
        def t_spike(s, k, v0, r0, lam):
            t1a = np.log(s.T@s / 2)
            t1b = np.log(r0*(s.T@s) - s.T@k / lam )
            t2 = ((s.T@k)/lam + r0*(s.T@s) + v0)
            t3 = lam + np.log(s.T@s / 2)
            
            print(t1a, t1b, t2, t3)
            
            return (t1a * t1b - t2)/t3
        
        k0 = 2
        
        print('\t\tPlotting Voltage Prediction\n')
        data = run_sim(N,  k = k0, D_scale=3, dt = dt)
        
        k = np.zeros((N,1))
        k[0] = k0
        
        s = np.zeros(k.shape)
        s[0] = data['D'][0,0]
        v0 = data['V'][0,0]
        r0 = data['r'][0,0]
        lam, _ = np.linalg.eig(data['A'])
        lam = lam[0]
        
        plt.figure()
        plt.loglog(data['t'], v(s, k, v0, r0, data['t'], lam), label='Derived Expression',linewidth=4)
        plt.loglog(data['t'], data['V'][0,:], 'x', label='Numerical Simulation')
        plt.axvline(x=t_spike(s, k, v0, r0, lam))
        plt.xlabel('Time (s)')
        plt.ylabel(r'Voltage')
        plt.title('Predited Neuron Voltage')
        plt.legend()
        plt.savefig(this_dir + '/' + 'predited_neuron_voltage.png', bbox_inches='tight')
      
    def plot_rate_vs_k():
        print('\t\tPlotting Rate vs Drive Strength k\n')
        ks = np.logspace(0, 4, num=10)
        rates =np.zeros(ks.shape)
        ss = np.zeros(ks.shape)
        for i, k in enumerate(ks):
            print('{0} / {1}'.format(i + 1, len(ks)))
            data = run_sim(N, 1, k = k, dt = dt)
            rates[i] = data['spike_nums'][0]   /  data['t'][-1]  
            ss[i] = (data['D'][0,0])
        
        print(phi(sk=ss/ks, lam=-1))
        
        plt.figure()
        plt.loglog(ks/ss, phi(sk=ss/ks, lam=-1), label='Derived Expression',linewidth=4)
        plt.loglog(ks/ss, rates, 'x', label='Numerical Simulation')
        plt.xlabel(r'Drive Strength Ratio $\frac{||S_1||}{||k||}$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate Response to Constant Driving')
        plt.legend()
        plt.savefig(this_dir + '/' + 'phi_const_driving.png', bbox_inches='tight')
    
    def plot_xhat_estimate_explicit():
        print('\t\tPlotting Network Estimate Comparison to Explicit Equation\n')
        
        def x_hat_explicit_phi(t, phi): 
            eq_pred_height = 1 / (1 - np.exp(-1/phi)) 
            return eq_pred_height * np.exp(- np.mod(t, 1/phi))
        
        def x_hat_explicit(t, s, k, l):
            p = phi(s/k, l)
            return x_hat_explicit_phi(t, p) 
            
            
        k = 1
        T = 10
        s = 1
        for k in np.logspace(0,2,num=10):
            data = run_sim(N, k = k, T =  T, dt=dt, stim='const')
            lam, _ = np.linalg.eig(data['lds'].A)
            lam = lam[0]
            idx = 0 
        
            plt.figure()
            plt.plot(data['t'],s * x_hat_explicit(data['t'] - data['O'][0,0], s,k,lam),'--', label='Derived Expression',linewidth=2)
            plt.plot(data['t'],data['x_hat'][idx,:],'r',label='Simulated Network Estimate',linewidth=2,alpha=.5)
            plt.plot(data['t'],data['x_true'][idx,:],'k', label = 'Target System', linewidth=2)                
            plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$)")
            plt.ylabel('Network Decode Dimension 0')
            plt.legend()
            plt.title('Predicted Network Decode Comparison')
        plt.savefig(this_dir + '/' + 'network_decode_long_term_estimate.png', bbox_inches='tight')

    def plot_per_spike_rmse_vs_k_phi():
        print('\t\tPlotting Per Spike RMSE vs k and phi')    
        T = 80
        
        ks = np.logspace(0.001,2, num = 50)
        ks_continuous = np.logspace(0.001,2, num = 1000)
        ss = np.zeros(ks.shape)

        rmses = np.zeros(ks.shape)
        rmse_stds = np.zeros(ks.shape)
        rates = np.zeros(ks.shape)
        
        for i,k in enumerate(ks):
            print('{0} / {1}'.format(i+1, len(ks)))
            data = run_sim(N, 1, k = k, T =  T,  dt = dt, D_scale=1) 
            
            rates[i] = data['spike_nums'][0]   /  data['t'][-1]  
            ss[i] = (data['D'][0,0])
            (mean, std) = per_spike_rmse_numerical(data, 0)
            rmses[i] = mean
            rmse_stds[i] = std       
                
        l, _ = np.linalg.eig(data['lds'].A)
        l = l[0]
  
        rates_continuous = np.logspace(-1,3,num=1000)
        
        plt.figure()
        plt.loglog(ks_continuous, rmse_sp_ksl(ks_continuous, 1, l), linewidth=4, label = 'Derived Expression')
        plt.loglog(ks, rmses, 'x', label='Numerical Simulation',linewidth=4,markersize=10)
        plt.errorbar(ks, rmses, yerr = rmse_stds, fmt = 'none')
        plt.xlabel(r'Drive Strength k')
        plt.ylabel(r'per-Spike RMSE')
        plt.legend()
        plt.title('Network Estimate RMSE per Spike vs Drive Strength k')
        plt.savefig(this_dir + '/' + 'rmse_sp_vs_ksl_const_driving.png', bbox_inches='tight')
        
        plt.figure()
        plt.loglog(rates_continuous, rmse_phi_const_driving(rates_continuous), linewidth=4, label = 'Derived Expression')
        plt.loglog(rates, rmses, 'x', label='Numerical Simulation',linewidth=4,markersize=10)
        plt.errorbar(rates, rmses, yerr = rmse_stds, fmt = 'none')
        plt.xlabel(r'Spike Rate ($\phi$)')
        plt.ylabel(r'per-Spike NRMSE')
        plt.legend()
        plt.title(r'Network Estimate NRMSE per Spike vs $\phi$')
        plt.savefig(this_dir + '/' + 'rmse_sp_vs_phi_const_driving.png', bbox_inches='tight')
        
    name = 'const_drive_strength'
    this_dir =  check_dir(name)
    
    print('\tPlotting Constant Driving Strength\n')
    
    
    run_plots()
             
def plot_const_dynamical_system(show_plots=True,N = 4, T = 10, dt = 1e-5):
   
    def run_plots():
        plot_rate_vs_s()
        #plot_xhat_estimate_explicit()
        #plot_per_spike_rmse_vs_phi_s()
        #plot_xhat_estimate_explicit()
 
    def plot_rate_vs_s():
        print('\t\tPlotting Rate Versus s\n')
        ss = np.logspace(-1, 1, num=10)
        ss_continuous = np.logspace(-1, 1, num=1000)
        rates = []
        plt.figure()
        for i, s in enumerate(ss):
            print('{0} / {1}'.format(i+1, len(ss)))
            data = run_sim(N, 1, k = 1, T = T, D_scale = s, D_type='simple', dt = dt)
            rates.append( data['spike_nums'][0]   /  data['t'][-1]  )
            
            #plt.plot(data['t'], data['x_hat'][0,:],label=s)
            #plt.legend()
        #plt.show()
        rates = np.asarray(rates)
        
        plt.figure()
        plt.loglog(1/ss_continuous, phi(ss_continuous, k=1), label='Derived Expression',linewidth=4)
        plt.loglog(1/ss, rates, 'x', label='Numerical Simulation')
        plt.xlabel(r'Drive Ratio $\frac{k}{S}$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate Response to Constant Driving')
        plt.legend()
        plt.savefig(this_dir + '/' + 'phi_vs_s_const_dynamical_system.png', bbox_inches='tight')
        
    def plot_xhat_estimate_explicit(): 
        print('\t\tPlotting Network Estimate Comparison to Explicit Equation\n')
        
        ss = np.logspace(-1,3, num = 10)
        idx =0 
        for i,s in enumerate(ss):
            print('Data point {0} / {1}'.format(i+1, len(ss)))
            data = run_sim(N, 1, k = 1, T =  T, D_scale= s, dt = 1e-4, stim='const' ) 
            
              
            plt.figure()
            plt.plot(data['t'],x_hat_explicit_s(data['t'] - data['O'][0,0], s),'--', label='Derived Expression',linewidth=2)
            plt.plot(data['t'],data['x_hat'][idx,:],'r',label='Simulated Network Estimate',linewidth=2,alpha=.5)
            plt.plot(data['t'],data['x_true'][idx,:],'k', label = 'Target System', linewidth=2)
            plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$)")
            plt.ylabel('Network Decode Dimension 0')
            plt.legend()
            plt.title('Predicted Network Decode Comparison')
            plt.savefig(this_dir + '/' + 'network_decode_long_term_estimate_const_dynamical_system.png', bbox_inches='tight')
        
    def plot_per_spike_rmse_vs_phi_s():
        print('\t\tPlotting Per Spike RMSE vs s and phi\n')    
        ss = np.logspace(-1,3, num = 50)
        T = 80
        
        rmses = []
        rmse_stds = []
        rates = []
        
        for i,s in enumerate(ss):
            print('Data point {0} / {1}'.format(i+1, len(ss)))
            data = run_sim(N, 1, k = 1, T =  T, D_scale= s, dt = 1e-4 ) 
            rates.append( data['spike_nums'][0]   /  data['t'][-1]  )
            (mean, std) = per_spike_rmse_numerical(data, 0)
            rmses.append(mean)
            rmse_stds.append(std)
            
            
        rmses = np.asarray(rmses)
        rates =  np.asarray(rates)        
        
        phis = np.logspace(-1,3, num=1000)
        
        plt.figure(figsize=(16,9))
        ss_continuous = np.logspace(-1,3, num = 1000)
    
        plt.loglog(ss_continuous, rmse_phi_s_const_dynamics(ss_continuous), linewidth=4, label = 'Derived Expression')
        plt.loglog(ss, rmses, 'x', label='Numerical Simulation',linewidth=4,markersize=10)
        plt.errorbar(rates, rmses, yerr = rmse_stds, fmt = 'none')
        plt.xlabel(r'Decoder Scale $S_1$')
        plt.ylabel(r'per-Spike RMSE')
        plt.legend()
        plt.title(r'Network Estimate RMSE per Spike vs $S_1$')
        plt.savefig(this_dir + '/' +'per_spike_rmse_vs_s_constant_dynamics', bbox_inches='tight')
        
        
        plt.figure(figsize=(16,9))
        plt.loglog(phis, rmse_phi_const_dynamics(phis), linewidth=4, label = 'Derived Expression')
        plt.loglog(rates, rmses, 'x', label='Numerical Simulation',linewidth=4,markersize=10)
        plt.errorbar(rates, rmses, yerr = rmse_stds, fmt = 'none')
        plt.xlabel(r'Spike Rate ($\phi$)')
        plt.ylabel(r'per-Spike RMSE')
        plt.legend()
        plt.title(r'Network Estimate RMSE per Spike vs $\phi$')
        plt.savefig(this_dir + '/' + 'per_spike_rmse_vs_phi_constant_dynamics.png', bbox_inches='tight')
        
     
    name = 'const_dynamical_system'
    this_dir =  check_dir(name)
    
    
    print('\tPlotting Constant Dynamical System\n')
    run_plots()
    

    if show_plots:
            plt.show()
    
def plot_pcf_gj_sc_comparison(show_plots=True,N = 32, T = 1000, dt = 1e-3, num_sim_points = 10):   
     
    def run_plots(num_sim_points = 10):   
        #plot_demos() 
        #plot_pcf_gj_sc_long_term_estimates_explicit() 
        #plot_pcf_gj_sc_rates(num_sim_points) 
        #plot_pcf_gj_sc_constant_stim_decode()
        #plot_pcf_gj_membrane_trajectories()
        plot_pcf_gj_sc_per_spike_rmse(num_sim_points)
        
    def pcf_estimate_explicit(d, t):
            phi = pcf_phi(d, 1) 
            eq_pred_height = d[0] / (1 - np.exp(-1/phi) )
            return eq_pred_height * (np.exp(- np.mod(t, phi**-1) ))  

    def plot_demos(): 
        drive_freq = .25
        drive_amp  = 10 
        A =  - np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([.5, .5])
        D = gen_decoder(A.shape[0], N, mode = '2d cosine')
        sin_func = lambda t :  drive_amp * np.asarray([np.cos( drive_freq * 2 * np.pi*t), np.sin( drive_freq * 2 * np.pi*t)]) + 8
        
        _, uA = np.linalg.eig(A)    
        sin_func = lambda t :  2 * uA[:,0]
         
        lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
        
        sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds)
        gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds)
        pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)
        
        sc_data = sc_net.run_sim() 
        gj_data = gj_net.run_sim()
        pcf_data= pcf_net.run_sim()
            
        models = [
            (sc_data, 'Self-Coupled'),
            (gj_data, 'Gap-Junction'),
            (pcf_data, 'PCF')
        ]
        
        for data, model_name in models:
                 
            plot_step = 10
            
            plt.figure()
            for i in range(4):
                plt.plot(data['t'],data['V'][i,:], label='Neuron %i Voltage'%i)
            plt.legend()
            plt.title(model_name + ' Neuron Membrane Potentials')
            plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
            plt.ylabel('Membrane Potential')
            plt.savefig(this_dir + '/' + 'demo_plot_membrane_potentials_' + model_name + '.png',bbox_inches='tight')
                    
            
            plt.figure()
            plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step],c='r',label='Decoded Network Estimate (Dimension 0)' )
            plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step],c='g',label='Decoded Network Estimate (Dimension 1)' )
            plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][0,0:-1:plot_step],c='k')
            plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][1,0:-1:plot_step],c='k',label='True Dynamical System')
            plt.title('Network Decode ' + model_name)
            plt.legend()
            plt.xlabel(r'Dimensionless Time $\tau_s$')
            plt.ylabel('Decoded State')
            plt.savefig(this_dir + '/' + 'demo_plot_network_decode_' + model_name + '.png',bbox_inches='tight')
            
            plt.figure()
            plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step] - data['x_true'][0,0:-1:plot_step],c='r',label='Estimation Error (Dimension 0)' )
            plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1,0:-1:plot_step] - data['x_true'][1,0:-1:plot_step],c='g',label='Estimation Error (Dimension 1)' )
            plt.title('Decode Error ' + model_name)
            plt.legend()
            plt.xlabel(r'Dimensionless Time $\tau_s$')
            plt.ylabel('Decode Error')
            plt.savefig(this_dir + '/' + 'demo_plot_decode_error_' + model_name + '.png',bbox_inches='tight')
            
            plt.figure()
            #cbar_ticks = np.round(np.linspace( start = np.min(data['V']), stop = .5,  num = 8, endpoint = True), decimals=1)
            plt.imshow(data['V'],vmax=np.max(data['V']), vmin=np.min(data['V']))
            plt.xlabel(r"Dimensionless Units of $\tau$")
            plt.axis('auto')
            #cbar = plt.colorbar(ticks=cbar_ticks)
            cbar = plt.colorbar()
            cbar.set_label('$v_j$')
            plt.title(model_name + ' Neuron Membrane Potentials')
            plt.ylabel('Neuron #')
            #plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])
            plt.savefig(this_dir + '/' + 'demo_plot_membrane_potential_image' + model_name + '.png',bbox_inches='tight')
            
        print('Figures Saved')
    
    def plot_pcf_gj_sc_rates(num_sim_points): 
        
        print('\t\tPlotting Rate Versus ||d_j||\n')
        
        ks = np.logspace(-3, 0,num=num_sim_points)
        ks_continuous = np.logspace(-3, 0, num = 1000 )
        pcf_rates = np.zeros((ks.shape))
        
        
        for i, k in enumerate(ks):
            print('{0} / {1}'.format(i+1, len(ks)))
            A =  - np.eye(2)
            B = np.eye(2)
            x0 = np.asarray([.5, 0])
            D = k * gen_decoder(A.shape[0], N, mode = '2d cosine')

            sin_func = lambda t :  np.asarray([1, 0])
            lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
            
            #sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds)
            #gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds)
            pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)
            
            #sc_data = sc_net.run_sim() 
            #gj_data = gj_net.run_sim()
            pcf_data= pcf_net.run_sim()
            
            pcf_jmax =np.argmax(pcf_data['spike_nums'])
            
            pcf_rates[i] = pcf_data['spike_nums'][pcf_jmax] / pcf_data['t'][-1]
            
        
        pcf_ds_continuous = np.zeros((D.shape[0], len(ks_continuous)))
        pcf_ds = np.zeros((D.shape[0], len(ks)))
        for i in range(pcf_ds.shape[1]):
            pcf_ds[:,i] = D[:,pcf_jmax] * ks[i]
        for i in range(pcf_ds_continuous.shape[1]):
            pcf_ds_continuous[:,i] = D[:,pcf_jmax] * ks_continuous[i]
            
             
            
        models = [
         #  (sc_data, 'Self-Coupled'),
         #   (gj_data, 'Gap-Junction'),
            (pcf_data, 'PCF', pcf_ds_continuous, pcf_ds)
        ]
                
        rate_funcs = {
                'PCF' : lambda k : pcf_phi(D[:,0], k)            
            }
        
        rate_measurements = {
            'PCF' : pcf_rates            
            }
        
        plt.figure()
        for _, model_name, ds_continuous, ds in models:
            plt.loglog(np.linalg.norm(ds_continuous,axis=0), rate_funcs[model_name](ks_continuous), label=model_name + ' Derived Expression',linewidth=4)
            plt.loglog(np.linalg.norm(ds,axis=0), rate_measurements[model_name], 'x', label=model_name + ' Numerical Simulation')
        plt.xlabel(r'Decoder Matrix Scale $||d_{j_{max}}||$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate Response to Constant Driving Strength')
        plt.legend()
        plt.savefig(this_dir + '/' + 'const_dynamics_' + model_name + '_rate_vs_d.png', bbox_inches='tight')
       
    def plot_pcf_gj_sc_long_term_estimates_explicit():  
        
                
        A =  - np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([.5, 0])
        
                
        
        sin_func = lambda t :  np.asarray([1, 0])
        
        lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
        
            
        ks = np.logspace(-3, 0,num=10)
        ks = [1]
        for i, k in enumerate(ks):
            print('{0} / {1}'.format(i+1, len(ks)))
            
            d = A.shape[0]
            
            _, uA = np.linalg.eig(A)
            
            D = k * uA @ np.hstack((
            np.eye(d),
            np.zeros((d, N - d))
            ))

            
            _, sD, _ = np.linalg.svd(D)
            
            sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds)
            pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)
            gj_net = GapJunctionDeneveNet(T, dt, N, D, lds, spike_trans_prob=1)
            
            
            sc_data = sc_net.run_sim() 
            pcf_data= pcf_net.run_sim()
            gj_data = gj_net.run_sim()
            
                
            models = [
                (sc_data, 'Self-Coupled',lambda t : x_hat_explicit_s(t, sD[0]), 'x', 'r', 20) , 
                (gj_data, 'Gap-Junction', lambda t : pcf_estimate_explicit(D[:,0], t),'d', 'g', 20),
                (pcf_data, 'PCF', lambda t : pcf_estimate_explicit(D[:,0], t), '.', 'b', 10)
            ]
        
            plt.figure()
    
            for data, model_name, estimation_func, marker, color, msize in models:
                     
                plot_step = 10
                
                plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0,0:-1:plot_step],marker, markersize = msize,  c=color,label=model_name + ' Network Estimate', alpha = 1)
                plt.plot(data['t_true'][0:-1:plot_step], estimation_func(data['t_true']- data['O'][0,0])[0:-1:plot_step],'--', c=color, alpha=.5)
                
            plt.title('Estimation of Network Decode (Dimension 0 )')
            plt.legend()
            plt.xlabel(r'Dimensionless Time $\tau_s$')
            plt.ylabel('Decoded State')
            plt.savefig(this_dir + '/' + 'const_dynamics_network_decode_comparison_sc_pcf_gj.png',bbox_inches='tight')           
    
    def plot_pcf_gj_sc_constant_stim_decode():
        A =  - np.eye(2)
        B = np.eye(2)
        D = gen_decoder(A.shape[0], N, mode = '2d cosine')
        x0 = np.asarray([.5, 0])
        sin_func = lambda t :  np.asarray([1, 0])
        
        lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
        
        sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds)
        gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds)
        pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)

        sc_data = sc_net.run_sim() 
        gj_data = gj_net.run_sim()
        pcf_data= pcf_net.run_sim()
            
        linestyles = {
            'Self-Coupled' : '-',
            'Gap-Junction' : '--',
            'PCF' : '-'
            }
            
        models = [
          (sc_data, 'Self-Coupled'),
          (pcf_data, 'PCF'),
          (gj_data, 'Gap-Junction')
            
        ]
        
        plt.figure()
        for data, model_name in models: 
            plot_step = 10
        
            plt.plot(
                data['t'][0:-1:plot_step],
                data['x_hat'][0,0:-1:plot_step],
                linestyles[model_name],
                linewidth = 2,
                label=model_name + '  Network Estimate (Dimension 0)' 
            )
        plt.title('Network Estimate Comparison')
        plt.legend()
        plt.xlabel(r'Dimensionless Time $\tau_s$')
        plt.ylabel('Decoded State')
        plt.savefig(this_dir + '/' + 'const_dynamics_network_decode_.png',bbox_inches='tight')           
    
    def plot_pcf_gj_membrane_trajectories():
        
        def v(d,t):
            return d[0] - np.exp(-t) * (d[0] + .5 * d[0]**2)
        
        A =  - np.eye(2)
        B = np.eye(2)
        D =  gen_decoder(A.shape[0], N, mode = '2d cosine')
        x0 = np.asarray([pcf_estimate_explicit(D[:,0], 0), 0])
        sin_func = lambda t :  np.asarray([1, 0])
        
        lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
        
        gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds)
        pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)
        
         
        
        pcf_data= pcf_net.run_sim()
        gj_data = gj_net.run_sim()
        
        j_max = np.argmax(pcf_data['spike_nums'])
        
        plt.figure()
        plot_step = 100
        ts = pcf_data['t'][0:-1:plot_step]
        plt.plot(ts, pcf_data['V'][j_max,0:-1:plot_step],label='PCF',linewidth=4)
        
        for i in [j_max]: #range(4):
            plt.plot(ts, gj_data['V'][i,0:-1:plot_step],'--',linewidth=4,label='gap-junction')
        plt.plot(ts, v(D[:,j_max], ts - gj_data['O'][j_max,0]), label='Derived interspike trajectory')
        plt.title('PCF and gap-junction membrane potentials')
        plt.legend()
        plt.xlabel(r'Dimensionless Time $\tau_s$')
        plt.ylabel('Membrane Potential of Spiking Neuron')
        plt.savefig(this_dir + '/' + 'const_dynamics_voltage_trajectory_gj_vs_pcf.png',bbox_inches='tight')
    
    def plot_pcf_gj_sc_per_spike_rmse(num_sim_points):
        def rmse_per_spike_pcf(phi):
            t1 = 4 * phi * np.tanh(.5 * phi**-1)
            t2 = (2 *phi)  * np.tanh(.5*phi**-1)**2 * (1 - np.exp(-2/phi))/ (1 - np.exp(-1/phi))**2
            
            return np.sqrt(1 - t1 + t2)
        
#             t1 = 1/phi
#             t2 = (2 * d[0] )
#             t3 =(d[0] / (1 - np.exp(-1/phi)) )**2 *  .5 * (1 - np.exp(-2/phi))
#             
#             return np.sqrt(phi * (t1 - t2 + t3))
#             
#             
#             d1 = 2 * (1 - np.exp(-1/phi)) / (1 + np.exp(-1/phi))
#             t1 =  phi * 2*d1*(1 + 1 / (2 * d1) ) * (1 - np.exp(-1/phi)) 
#             t2 =  phi * (d1**2 / 2) * (1 + 1/d1 + (.25) * d1**-2 ) * (1 - np.exp(-2 / phi))
#             
#             return np.sqrt(1 - t1 + t2)
#         
        
        print('\t\tPlotting Rate Versus per-spike RMSE\n')
        A =  - np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([.5, 0])
        
        ks = np.logspace(-4, 4,num=num_sim_points)
       
        pcf_rates = np.zeros(ks.shape)
        pcf_rmses_numerical = np.zeros(ks.shape)
        pcf_rmses_derived = np.zeros(ks.shape)

                
        gj_rates = np.zeros(ks.shape)
        gj_rmses_numerical = np.zeros(ks.shape)
        gj_rmses_derived = np.zeros(ks.shape)
        
        sc_rates = np.zeros(ks.shape)
        sc_rmses_numerical = np.zeros(ks.shape)
        sc_rmses_derived = np.zeros(ks.shape)
        
        for i, k in enumerate(ks):
            print('{0} / {1}'.format(i+1, len(ks)))
            sin_func = lambda t :  np.asarray([1, 0])
            lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
            
            d = A.shape[0]
            _, uA = np.linalg.eig(A)
            
            D = k * uA @ np.hstack((
            np.eye(d),
            np.zeros((d, N - d))
            ))
            
            D = k * gen_decoder(A.shape[0], N, mode = '2d cosine')
            
            sc_net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds)
            gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds)
            pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=1)
            
            sc_data = sc_net.run_sim() 
            gj_data = gj_net.run_sim()
            pcf_data= pcf_net.run_sim()
            
            pcf_jmax =np.argmax(pcf_data['spike_nums'])
            gj_jmax = np.argmax(gj_data['spike_nums'])
            sc_jmax = 0
            
            pcf_rates[i] = pcf_data['spike_nums'][pcf_jmax] / pcf_data['t'][-1]
            gj_rates[i] = gj_data['spike_nums'][gj_jmax] / gj_data['t'][-1]
            sc_rates[i] = sc_data['spike_nums'][sc_jmax] / gj_data['t'][-1]
            
            pcf_rmses_numerical[i] = per_spike_rmse_numerical(pcf_data, pcf_jmax)[0]
            pcf_rmses_derived[i] = rmse_per_spike_pcf(pcf_rates[i])
             
            gj_rmses_numerical[i] =  per_spike_rmse_numerical(gj_data, gj_jmax)[0]
            gj_rmses_derived[i] =  rmse_per_spike_pcf(gj_rates[i])
            
            sc_rmses_numerical[i] = per_spike_rmse_numerical(sc_data, gj_jmax)[0]
            sc_rmses_derived[i] = rmse_phi_const_dynamics(sc_rates[i])
            
        models = [
            (sc_data, 'Self-Coupled', 'x', 'r', 20),
            (gj_data, 'Gap-Junction', 'd', 'g', 20),
            (pcf_data, 'PCF', '.', 'b', 10)
        ]
                
        rmses_derived = {
                'PCF' : pcf_rmses_derived,
                'Gap-Junction' : gj_rmses_derived,    
                'Self-Coupled' : sc_rmses_derived 
            }
        
        rate_measurements = {
            'PCF' : pcf_rates,
            'Gap-Junction' : gj_rates,    
            'Self-Coupled' : sc_rates        
            }
        
        rmses_numerical = {
            'PCF' : pcf_rmses_numerical,
            'Gap-Junction' : gj_rmses_numerical,    
            'Self-Coupled' : sc_rmses_numerical      
            }
        
        plt.figure()
            
        for _, model_name, marker, color, msize in models:
            plt.loglog(rate_measurements[model_name], rmses_derived[model_name], '--','k',c=color, alpha = .5)
            plt.loglog(rate_measurements[model_name], rmses_numerical[model_name], marker=marker, c=color, markersize= msize, label=model_name)
        plt.ylabel(r'per-Spike RMSE')
        plt.xlabel(r'Neuron Firing Rate $\phi$')
        plt.title('per-Spike RMSE vs Neuron Firing Rate')
        plt.legend()
        plt.savefig(this_dir + '/' + 'per_spike_rmse_vs_phi.png', bbox_inches='tight')
        
    name = 'pcf_gj_sc_comparison'
    this_dir =  check_dir(name)
    
    
    run_plots(num_sim_points)
    
    
    #plot_pcf_gj_sc_per_spike_rmse(20) 
    if show_plots:
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
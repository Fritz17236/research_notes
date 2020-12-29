'''
Plots for HSF Research
'''

from matplotlib import cm
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from scipy.special import lambertw

# Add Base Directory for HSFNets and utils packages here.

path_to_Misc_Research_Code = '/home/chris/Desktop/git_repos/Misc-Research-Code'

 
print(path_to_Misc_Research_Code)
import sys
sys.path.append(path_to_Misc_Research_Code)
import os

import matplotlib.patches as patches
import matplotlib

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
plt.rcParams['lines.linewidth'] = 3

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
  
def phi_lam0(bk, y, s):
        t1 = 1 / ( 2 * bk) * (s - 2*y)
        arg = (y/bk) * np.exp(- (s/2 - y)/bk)
        t2 =  t1 + lambertw(arg)
        return 1/np.real(t2)
  
def sc_phi(bk, s, l):

     #top = 2 * bk + np.abs(l) * s
     #bot = 2 * bk - np.abs(l) * s

     #return 1 / np.log(top / bot)

    t1 = 1 / (bk/(s*np.abs(l))-0.5)
    return 1/np.log(1 + t1)

def pcf_phi(k, s, l):
    return sc_phi(k, s, l)
    
def phi_bk(s, bk):
    result = np.log( (2*bk + s) / (2*bk - s))**-1
    
    
    return result

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


    ep= np.exp(-1/phi)
    ep2 = np.exp(-2/phi)






    return np.sqrt( (4*ep - 2*ep2 - 1) / (4*(ep-1)**2))

def rmse_phi_const_dynamics(phi):
    t1 =  np.tanh(
            1 / (2 * phi )
         )
    return np.sqrt(1 - 2 * phi * t1)


# Functions for PCF Network

def per_spike_rmse_numerical(data, idx):
    ''' given data from a network sim, compute the average per-spike rmse of neuron indexed by idx'''

    num_spikes = data['spike_nums'][idx]

    # se = np.square(data['error'][idx,:])
    # delta_t = data['dt']
    # mse = delta_t * np.sum(se) / (data['t'][-1] - data['t'][0])
    # return np.sqrt(mse)

    spikes = data['O'][idx,:num_spikes+1]
    
    if num_spikes < 6 :
        print('only {0} spikes occurred'.format(num_spikes))
        return np.nan, np.nan
    
    rmses = []
    
    #num_trials = int(num_spikes//1)
    num_trials = num_spikes
    delta_t = data['dt']

    start_spike = num_spikes-num_trials
    for trial in np.arange(start_spike, num_spikes - 1):
        try:
            left_idx = np.argwhere(spikes[trial] < data['t'][:])[0][0]
            right_idx = np.argwhere(spikes[trial + 1] <  data['t'][:])[0][0]
        except Exception as e:
            return np.nan, np.nan
    
    

        #e = data['x_hat'][0,left_idx:right_idx] - data['x_true'][0,left_idx:right_idx]
        se = np.square(data['error'][0,left_idx:right_idx]) 
    
        
        mse =  delta_t * np.sum(se)  / (data['t'][right_idx]-data['t'][left_idx])
        rmses.append(np.sqrt(mse))
    return np.mean(np.asarray(rmses)), np.std(np.asarray(rmses))

def run_sim(N, p=1, T = 20,  k = 1, dt = 1e-5, stim='const', D_scale = 1, D_type='none'):

    A =  - .01*np.eye(2)
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
def plot_raster(data):
    '''
    Given a simulation data of N neurons,
    plot the spike raster. caller must call plt.show() afterward
    '''
    plt.figure()

    for j in range(data['N']):
        #nonzero spikes
        spike_times = data['O'][j,:]
        spike_times = spike_times[spike_times != 0]
        nums = j * np.ones(spike_times.shape) + 1
        plt.scatter(spike_times, nums,  c = 'k',marker='.')

        plt.xlabel('Time (dimensionless)')
        plt.ylabel('Neuron Number')
        plt.ylim([.5, data['N'] + .5])
        plt.yticks(ticks=np.arange(data['N']) + 1, labels=np.arange(data['N']) + 1)
        plt.title('Spike Raster')

def plot_test(show_plots=True,N = 4, k = 1, T = 10, dt = 1e-5):
    def get_steady_state_sequence(k, D, x0, num_iter=30):
        count = 1
        sequence = []
        while count < num_iter:
            dtx = D.T @ x0
            e_sps= np.zeros((len(dtx,)))
            for i in np.arange(len(dtx)):
                dtk_d2 = D[:,i].T @ k - .5 * (D[:,i].T @ D[:,i])
                e_sps[i] =  dtx[i] / dtk_d2


            e_sps[np.isclose(e_sps, 1) ] = np.inf
            e_sps[e_sps <= 1 ] = np.inf

            spike_idx = np.argmin(e_sps)
            print(spike_idx)

            sp_time = np.log(e_sps[spike_idx])

            assert (sp_time >= 0), "Spike {3} time {0} not positive,  xhat = {1}, e_sps={2}".format(sp_time, x0, e_sps[spike_idx], spike_idx)

            x0 = x0 * np.exp(-sp_time) + D[:,spike_idx]

            sequence.append((spike_idx, sp_time, x0))
            count += 1

        return sequence

    A = -np.eye(2)
    B = -A

    D = gen_decoder(A.shape[0], N,mode='2d cosine')
    #D = gen_decoder(A.shape[0], N)

    D = .5 * D

    theta  = 80 * (np. pi / 180)  # in degrees

    k = np.asarray([np.cos(theta), np.sin(theta)])
    k /= 1 * np.linalg.norm(k)

    sin_func = lambda t: k
    x0 = k #np.asarray([-1, 1])

    sequence = get_steady_state_sequence(sin_func(0), D, x0.copy(), num_iter = 30)

    plt.figure()
    for j in range(N):
        plt.scatter(D[0,j], D[1,j],label='Neuron {0}'.format(j + 1))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.legend()

    lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)
    #net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, lam_v=0)
    net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    data = net.run_sim()

    plot_step = 10

    plot_raster(data)
    plt.title(r"Spike Raster, $\theta$ = {0} deg".format(theta * (180 / np.pi)))


    vmin = -np.min(data['vth'])
    vmax = np.max(data['vth'])
    #
    plt.figure()
    for i in range(N):
        plt.plot(data['t'], data['V'][i, :], label='Neuron %i Voltage' % i)
        plt.axhline(y = data['vth'][i],color='k')


    elapsed = 0
    for idx, time, x_hat in sequence:
        elapsed += time
        plt.axvline(x=elapsed, ls='--',color='k')


    plt.legend()
    plt.title("Neuron Membrane Potentials")
    plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
    plt.ylabel('Membrane Potential')
    #
    #
    # plt.figure()
    # for j in range(N):
    #     plt.plot(data['t'], data['r'][j,:],label="neuron {0}".format(j))
    #
    # plt.figure()
    #
    #
    # num_spikes = np.max(data['spike_nums'])
    # #plt.imshow(data['O'][:,:num_spikes]!=0,interpolation='none')
    #
    # last_spikes = np.zeros((N,))
    #
    # for j in range(N):
    #     last_spikes[j] = data['O'][j, data['spike_nums'][j]-1]
    #
    # plt.scatter(np.arange(N)+1, last_spikes)
    #


    #cbar_ticks = np.linspace(start=vmin, stop=vmax, num=8, endpoint=True)
    #plt.imshow(data['V'], extent=[0, data['t'][-1], 0, 3], vmax=vmax, vmin=vmin)
    #plt.xlabel(r"Dimensionless Units of $\tau$")
   # plt.axis('auto')
    #cbar = plt.colorbar(ticks=cbar_ticks)
    #cbar.set_label(r'$\frac{v_j}{v_{th}}$')
    #cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
    #plt.title('Neuron Membrane Potentials')
    #plt.ylabel('Neuron #')
    #plt.yticks([.4, 1.15, 1.85, 2.6], labels=[1, 2, 3, 4])

    plt.figure()
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0, 0:-1:plot_step], c='r',
             label='Decoded Network Estimate (Dimension 0)')
    plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1, 0:-1:plot_step], c='g',
              label='Decoded Network Estimate (Dimension 1)')
    plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][1, 0:-1:plot_step], c='k')
    plt.plot(data['t_true'][0:-1:plot_step], data['x_true'][0, 0:-1:plot_step], c='k', label='True Dynamical System')
    plt.title('Network Decode')
    plt.legend()
    plt.xlabel(r'Dimensionless Time $\tau_s$')
    plt.ylabel('Decoded State')

    print(data['O'][:,0:20])
    elapsed = 0

    print([(i, j) for i ,j ,k in sequence])
    for idx, time, x_hat in sequence:
        elapsed += time
        plt.scatter(elapsed, x_hat[0],c='r',marker='x')
        plt.scatter(elapsed, x_hat[1],c='g', marker='x')

    # plot sequence

    # one axis is time to spike
    # other axis is neuron number
    sequence = get_steady_state_sequence(sin_func(0), D, x0.copy(), num_iter=1000)
    plt.figure()
    plt.ylabel("Time to Spike (dimensionless)")
    plt.xlabel("Neuron Number")

    for idx, time, x_hat in sequence:
        plt.scatter(idx + 1, time, c='k',marker='.')
    plt.xticks(ticks=np.arange(N)+1)
    plt.xlim([.5, N+.5])
    plt.ylim([0, 1])

    plt.figure()

    ts = [time for idx, times, x_hat in sequence]
    elapsed = 0
    for i, t in enumerate(ts):
        elapsed += t
        ts[i] = elapsed


    x_hats = [x_hat for idx, times, x_hat in sequence]
    x_hats = np.asarray(x_hats)
    cmap = cm.get_cmap('inferno', lut=len(ts))

    plt.scatter(x_hats[:,0], x_hats[:,1], c=ts, cmap=cmap)
    plt.scatter(k[0],k[1], marker='x')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.ylabel("Dimension 2")
    plt.xlabel("Dimension 1")


    # plt.figure()
    # plt.plot(data['t'][0:-1:plot_step], data['x_hat'][0, 0:-1:plot_step] - data['x_true'][0, 0:-1:plot_step], c='r',
    #          label='Estimation Error (Dimension 0)')
    # plt.plot(data['t'][0:-1:plot_step], data['x_hat'][1, 0:-1:plot_step] - data['x_true'][1, 0:-1:plot_step], c='g',
    #          label='Estimation Error (Dimension 1)')
    # plt.title('Decode Error')
    # plt.legend()
    # plt.xlabel(r'Dimensionless Time $\tau_s$')
    # plt.ylabel('Decode Error')

    plt.show()

def plot_complex_eigvals(show_plots=True, N=4, k=1, T=10, dt=1e-5):
    #A0 = -np.zeros((3,3))
    A0 = np.zeros((2, 2))
    A0[0,1] = -1
    A0[1,0] = 1
    A, P = real_jordan_form(A0)
    B = 0*np.eye(A.shape[0])
    x0 = np.asarray([1, 0])
    input = lambda t : np.ones(x0.shape)
    D = gen_decoder(A.shape[0], N, mode='2d cosine')
    #D = .1 * gen_decoder(A.shape[0], N)

    lds = sat.LinearDynamicalSystem(x0, A, B, u=input, T=T, dt=dt)

    # gjNet = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    # dNet = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, lam_v=0)
    # net = SecondOrderSymSCNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    #net = SecondOrderSCNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    #net = CSelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    #data = net.run_sim()
    # gjData= gjNet.run_sim()
    # dData = dNet.run_sim()
    num_pts = 30
    ss = np.logspace(-4,0,num=num_pts)
    rates_sc = np.zeros((num_pts,))
    rmses_sc = np.zeros((num_pts,))
    rates_gj = np.zeros((num_pts,))
    rmses_gj = np.zeros((num_pts,))
    rates_dn = np.zeros((num_pts,))
    rmses_dn = np.zeros((num_pts,))


    def rmse_numerical(data):
        ''' compute rmse averaged over simulation duration'''
        e = data['x_true'] - data['x_hat']
        se = [e[:,i].T @ e[:,i] for i in range(len(data['t']))]
        assert(len(se)==len(data['t']))
        mse = (dt / T) * np.sum(se)
        return np.sqrt(mse)


    for i, s in enumerate(ss):
        print("{0} / {1}".format(i + 1, num_pts ))
        Ds = s * D
        net = SecondOrderSymSCNet(T=T, dt=dt, N=N, D=Ds, lds=lds, t0=0)
        gjNet = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=Ds, lds=lds, t0=0)
        dNet = ClassicDeneveNet(T=T, dt=dt, N=N, D=Ds, lds=lds, t0=0, lam_v=0)
        gjData = gjNet.run_sim()
        dData = dNet.run_sim()
        data = net.run_sim()
        rates_sc[i] = np.sum(data['spike_nums']) / T
        rmses_sc[i] = rmse_numerical(data)

        rates_gj[i] = np.sum(gjData['spike_nums']) / T
        rmses_gj[i] = rmse_numerical(gjData)

        rates_dn[i] = np.sum(dData['spike_nums']) / T
        rmses_dn[i] = rmse_numerical(dData)





    plt.figure("rmses")
    plt.loglog(rates_sc, rmses_sc,c='r', label=r'Second Order SC', alpha =.5)
    plt.loglog(rates_gj, rmses_gj,c='g', label=r'GJ')
    plt.loglog(rates_dn, rmses_dn,c='b', label=r'Deneve')

    #plt.loglog(rates_sc, 1 / rates, label=r'$\frac{1}{\phi}$')
    #plt.loglog(rates_sc, 1 / np.square(rates), label=r'$\frac{1}{\phi^2}$')
    plt.xlabel(r"$\phi$")
    plt.legend()
    plt.ylabel(r"RMSE")
    #plt.axhline(dt)


    #
    # plt.figure('rates')
    # plt.loglog(ss, rates,label='total spike / T')
    # plt.loglog(ss, 1/ss,label='1/s')
    # plt.loglog(ss, 1 / (ss*ss), label='(1/s)^2')
    # plt.xlabel("s")
    # plt.legend()
    # plt.ylabel(r"$\phi$")

    plt.show()

    #
    #
    #
    # assert(False)
    plot_step = 10
    vmin = -np.min(data['vth'])
    vmax = np.max(data['vth'])
    ts = data['t']
    dim = data['dim']


    def instantaneous_spike_rate(data, idx,  t_width):
        ''' Given a dataset with neuron index idx, compute the instantaneous spike rate over a width t_width'''
        N = int(t_width / dt)
        rates = np.zeros((len(data['t']),))
        n_spikes = data['spike_nums'][idx]
        spike_train = data['O'][idx, 0:n_spikes]
        for j in np.arange(N, len(rates), step = 1):
            curr_time = data['t'][j]
            lookback = data['t'][j - N]
            rates[j] = np.sum([1 for t_spike in spike_train if t_spike <= curr_time and t_spike >= lookback])

        return rates / t_width
    # from j = N to len(T):
        # rate[j] = num_spikes / t_width

    rates = instantaneous_spike_rate(data, 0, 10)
    plt.plot(data['t'], rates)
    plt.plot(data['t'][1:],np.diff(rates))




    def u_pred(data, idx):
        '''given simualation data, predict the trajectory of tilde{u}_j where j=idx '''
        t = data['t']
        u_true = data['us'][idx,:]
        u0 = u_true[0]

        # get array of spike times
        n_spikes = data['spike_nums'][idx]
        spike_train = data['O'][idx, 0:n_spikes]
        u_predicted = np.zeros(t.shape)
        for t_spike in spike_train:
            spike_trajectory =  np.exp(-t + t_spike)
            spike_trajectory[t < (t_spike - data['dt'])] = 0
            u_predicted += spike_trajectory
        return u_predicted + np.exp(-t) * u0


    def rho_pred(data, idx):
        '''
        given simulation data, predict trajectory of rho_j where j = idx
        '''
        t = data['t']
        u0 = data['us'][idx, 0]
        rho0 = data['r'][idx, 0]
        n_spikes = data['spike_nums'][idx]
        spike_train = data['O'][idx, 0:n_spikes]
        t1 = spike_train[0]
        rho_predicted = np.zeros(t.shape)

        # index of time when spike occurs
        spike_idxs = np.squeeze(np.asarray([np.nonzero(np.isclose(t_spike, t)) for t_spike in spike_train]))
        u_preds = u_pred(data, idx)
        for idx_spike, t_spike in enumerate(spike_train):
            if idx_spike == 0:
                t_last_spike = 0
                t_idx_last_spike = 0
            else:
                t_last_spike = spike_train[idx_spike - 1]
                t_idx_last_spike = spike_idxs[idx_spike - 1]
            t_window = (t - t_last_spike)
            t_window[t < t_last_spike] = 0
            t_window[t > t_spike] = t_spike - t_last_spike
            up_last_spike = u_preds[t_idx_last_spike]

            term = t_window * np.exp(-t + t_last_spike) * up_last_spike
            rho_predicted += term

        t_win_1 = t.copy()
        t_win_1[t > t1] = t1
        init_term = np.exp(-t) * (rho0)
        rho_predicted += init_term




        #pred[t > t1] = pred[t > t1] -  t[t > t1] * u0
        return rho_predicted
        # for t_spike in spike_train[1,:]:
        #
        #     spike_trajectory = np.exp(-t + t_spike)
        #     spike_trajectory[t < (t_spike - data['dt'])] = 0
        #     u_predicted += spike_trajectory
        # return u_predicted + np.exp(-t) * u0


    plt.figure("u-prediction")
    plt.plot(data['t'], u_pred(data, 0),'--', c='r',label='predicted')
    plt.plot(data['t'], data['us'][0,:], c='g',alpha=.5,label='actual')
    plt.legend()

    plt.figure("rho-prediction")
    plt.plot(data['t'], rho_pred(data, 0),'--',c='r',label='predicted')
    plt.plot(data['t'], data['r'][0, :], c='g', alpha=.5,label='actual')
    plt.legend()
    plt.show()

    assert(False)


    # plot error trajectory as scatter, color with time
    # plt.figure("error_trajectory")
    # num_pts = 30000
    # e = (data['x_hat']- data['x_true'])
    # e = np.diff(e) / data['dt']
    # cmapsc = cm.get_cmap('Reds', lut=len(ts))
    # mask = np.arange(0, num_pts)
    # plt.scatter(e[0, mask], e[1, mask], c=ts[mask], marker='s', cmap=cmapsc)
    # cbar = plt.colorbar()
    # plt.title('Error Trajectory in State Space')
    # plt.xlabel(r'$\hat{x}_0$')
    # plt.ylabel(r'$\hat{x}_1$')
    # s = data['s']
    # rect = patches.Rectangle((-s / 2, -s / 2), s, s, linewidth=2, edgecolor='r',
    #                          facecolor='none',label=r" $\pm \frac{s}{2}$")
    # plt.gca().add_patch(rect)
    # cbar.set_label("Time Elapsed")
    # plt.xlim([-.5, .5])
    # plt.ylim([-.5, .5])
    # plt.legend()

    cols = cm.get_cmap('plasma',lut=N)
    for i in range(N):
        if i % 2 == 1:
            plt.figure("us")
            plt.plot(ts, data['us'][i,:],label='neuron %i'%i,linestyle='--')
            plt.ylabel(r"$u(\xi)$")
            plt.title(r"$u(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.legend()

            plt.figure("rs")
            plt.plot(ts, data['r'][i, :], label='neuron %i' % i,linestyle='--')
            plt.ylabel(r"$r(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$r(\xi)$")
            plt.legend()

            plt.figure("[Ca]")
            plt.plot(ts, data['V'][i, :], label='neuron %i' % i,linestyle='--')
            plt.ylabel(r"$[Ca](\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$[Ca](\xi)$")
            plt.legend()

            plt.figure(r"$V(\xi)$")
            plt.plot(ts, data['v_dots'][i, :], label='neuron %i' % i,linestyle='--',c=cols(i/N))
            plt.ylabel(r"$V(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$V(\xi)$")
            plt.legend()
            plt.axhline(data['vth'][i],c=cols(0/N))
            plt.axhline(-data['vth'][i], c=cols(0/N))

        else:
            plt.figure("us")
            plt.plot(ts, data['us'][i,:],label='neuron %i'%i,alpha=.5)
            plt.ylabel(r"$u(\xi)$")
            plt.title(r"$u(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.legend()

            plt.figure("rs")
            plt.plot(ts, data['r'][i, :], label='neuron %i' % i,alpha=.5)
            plt.ylabel(r"$r(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$r(\xi)$")
            plt.legend()

            plt.figure("[Ca]")
            plt.plot(ts, data['V'][i, :], label='neuron %i' % i,alpha=.5)
            plt.ylabel(r"$[Ca](\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$[Ca](\xi)$")
            plt.legend()

            plt.figure(r"$V(\xi)$")
            plt.plot(ts, data['v_dots'][i, :], label='neuron %i' % i,alpha=.5,c=cols(i/N))
            plt.ylabel(r"$V(\xi)$")
            plt.xlabel(r"$\xi$")
            plt.title(r"$V(\xi)$")
            plt.legend()

    #cmapsc = cm.get_cmap('Reds', lut=len(ts))
    #cmapgj = cm.get_cmap('Greens', lut=len(ts))
    #cmapsc2 = cm.get_cmap('Blues', lut=len(ts))
    #mask = np.arange(1, int(len(ts)//4))
    #plt.show()
    #plt.scatter(e[0, mask], e[1, mask], c=ts[mask], marker='s', cmap=cmapsc)
    #cbar = plt.colorbar()
    #plt.scatter(gj_e[0, mask], gj_e[1, mask], c=gj_ts[mask], marker='.', cmap=cmapgj)
    #sD = np.diag(data['sD']) * 1.01
    #plt.title('Error Trajectory in State Space')
    #plt.xlabel(r'$\hat{x}_0$')
    #plt.ylabel(r'$\hat{x}_1$')
    # for t in ts:

    #     sD = np.diag(data['sD'])
    #rect = patches.Rectangle((-sD[0, 0] / 2 , -sD[1, 1] / 2 ), sD[0, 0], sD[1, 1], linewidth=2, edgecolor='r', facecolor='none')
    #     rect2 = patches.Rectangle((- np.sqrt(2) * sD[1, 1] / 2, -np.sqrt(2) * sD[1, 1] / 2), np.sqrt(2) * sD[1, 1], np.sqrt(2) * sD[1, 1], linewidth=2, edgecolor=cmapsc2(t / ts[-1]), facecolor='none')
    #     circ = patches.Circle((0,0), radius=np.sqrt(sD[0,0])/2, facecolor='none',edgecolor='b')
    #     t=matplotlib.transforms.Affine2D().rotate_deg_around(0, 0, angle)
    #     t2 = matplotlib.transforms.Affine2D().rotate_deg_around(0, 0, -angle)
    #     rect.set_transform(t + plt.gca().transData)
    #     rect2.set_transform(t2 + plt.gca().transData)
    #plt.gca().add_patch(rect)
    #     plt.gca().add_patch(rect2)
    #     plt.gca().add_patch(circ)
    #
    #plt.xlim([-.5, .5])
    #plt.ylim([-.5 ,.5])
    #cbar.set_label('Time Elapsed')
    #data['V'] = np.real(data['V'])
    # plt.figure()
    # for i in range(N):
    #     plt.plot(data['t'], (data['v_dots'][i, :]), label='Neuron %i Voltage' % i)
    #     #plt.plot(data['t'], np.max(data['V'][i,:])/(2*np.pi) * np.unwrap(2*np.pi*data['V'][i, :]/np.max(data['V'][i,:]),discont = .1), label='Neuron %i Unwrapped' % i)
    #     #e = np.real((uAi @ data['x_true'] - data['x_hat'])[i,:])
    #     #em = np.max(e)
    #
    #     #plt.plot(data['t'], (em / (2 * np.pi)) * np.unwrap(2 * np.pi * (e / em), discont=.1), label='Network Error%i'%i)


    plt.figure()
    cbar_ticks = np.linspace( start = vmin, stop = vmax,  num = 8, endpoint = True)
    plt.imshow((data['v_dots']),extent=[0,data['t'][-1], 0,3],vmax=vmax, vmin=vmin,interpolation='none')
    plt.xlabel(r"Dimensionless Units of $\tau$")
    plt.axis('auto')
    cbar = plt.colorbar(ticks=cbar_ticks)
    cbar.set_label(r'$\frac{v_j}{v_{th}}$')
    cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
    plt.title(r'$V(\xi)$')
    plt.ylabel('Neuron #')
    plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])

    plt.figure()
    cbar_ticks = np.linspace(start=vmin, stop=vmax, num=8, endpoint=True)
    plt.imshow((data['V']), extent=[0, data['t'][-1], 0, 3], interpolation='none')
    plt.xlabel(r"Dimensionless Units of $\tau$")
    plt.axis('auto')
    cbar = plt.colorbar(ticks=cbar_ticks)
    #cbar.set_label(r'$$')
    cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
    plt.title(r'$[Ca](\xi)$')
    plt.ylabel('Neuron #')
    plt.yticks([.4, 1.15, 1.85, 2.6], labels=[1, 2, 3, 4])


    plt.figure()
    ts = data['t'][0:-1:plot_step]
    true = data['x_true'][0, 0:-1:plot_step]
    plt.plot(ts,  ( gjData['x_hat'])[0, 0:-1:plot_step], c='g', label='GJ Net')
    plt.plot(ts, (dData['x_hat'])[0, 0:-1:plot_step], c='b', label='Deneve Net')
    plt.plot(ts, (data['x_hat'])[0, 0:-1:plot_step], c='r', label='Second Order Net')
    # plt.plot(ts, (P @ data['x_hat'])[2, 0:-1:plot_step], c='b', label='Dimension 2')
    #plt.plot(ts, ( data['x_true'])[0, 0:-1:plot_step], c='k', label='True Dynamical System')
    plt.plot(ts, true, c='k', label='True Dynamical System')
    #plt.plot(ts, ( data['x_true'])[1, 0:-1:plot_step], c='k')
    plt.title('Network Decode')
    plt.legend()
    plt.ylim([-2, 2])
    plt.xlabel(r'$\xi$')
    plt.ylabel('Decoded State')


    plt.figure()
    plt.plot(ts,
             ( gjData['x_hat'])[0, 0:-1:plot_step] -(true),
             c='g',
             label='Gap Junction')
    plt.plot(ts,
             (dData['x_hat'])[0, 0:-1:plot_step] - (true),
             c='b',
             label='Deneve')
    plt.plot(ts,
             ( data['x_hat'])[0, 0:-1:plot_step] - (true),
             c='r',
             label='Second Order ')
    plt.title('Decode Error')
    plt.legend()
    plt.ylim([-2, 2])
    plt.xlabel(r' $\xi$')
    plt.ylabel('Decode Error')

    # plt.figure('edot')
    # plt.plot(ts,
    #          (1 / dt) * np.squeeze(np.diff((data['x_hat']-true)[0:-1:plot_step])),
    #          c='r',
    #          label='Second Order ')
    # plt.axhline(y=-data['s']/2,c='k',label='+/- vth')
    # plt.axhline(y=data['s']/2,c='k')
    #
    # plt.title(r'$\dot{\epsilon}$')
    # plt.legend()
    # plt.ylim([-2, 2])
    # plt.xlabel(r' $\xi$')
    # plt.ylabel(r'$\dot{\epsilon}$')

    # plt.figure("Mv")
    # plt.imshow(data["Mv"], norm=colors.SymLogNorm(1))
    # cbar = plt.colorbar()
    # plt.title("Voltage Coupling")
    # cbar.set_label("Coupling Strength")
    #
    #
    # plt.figure("MCa")
    # plt.imshow(data["MCa"],norm=colors.SymLogNorm(1))
    # cbar = plt.colorbar()
    # cbar.set_label("Coupling Strength")
    # plt.title("Calcium Coupling")
    #
    # plt.figure("Mo")
    # plt.imshow(data["Mo"])
    # cbar = plt.colorbar()
    # cbar.set_label("Coupling Strength")
    # plt.title("Voltage Fast Coupling")
    #
    # plt.figure("Mr")
    # plt.imshow(data["Mr"])
    # cbar = plt.colorbar()
    # cbar.set_label("Coupling Strength")
    # plt.title("Voltage Slow Coupling")


    #plot isi vs t
    # n_spikes = data['spike_nums'][0]
    # #isis = np.diff(data['O'][0,0:n_spikes])
    # plt.figure("ttls")
    # ts = data['t'] + data['O'][0,0]
    # ttls = np.zeros(ts.shape)
    # spike_train = data['O'][0, 0:n_spikes-1]


    # for i in range(N):
    #     n_spikes = data['spike_nums'][i]
    #     spike_train = data['O'][i, 0:n_spikes]
    #
    #     for j, t in enumerate(ts):
    #         try:
    #             last_spike = np.max(spike_train[spike_train <= t])  # get the greatest spike time less than current
    #             ttls[j] = t - last_spike
    #         except Exception as e:
    #             ttls[j] = 0
    #
    #     if i==0:
    #         plt.figure("ttls")
    #         plt.plot(ts, ttls, label='neuron {0}'.format(i) )
    #         plt.title("Time Since Last Spike")
    #         plt.xlabel(r"$\xi$")
    #         plt.ylabel("Time Since Last Spike")
    #         plt.legend()
    #
    #     num_spikes = [np.sum(spike_train[spike_train <= t] !=0 ) for j,t in enumerate(ts)]
    #     plt.figure("sat's")
    #     plt.ylabel("Number of Spikes")
    #     plt.xlabel(r"$\xi$")
    #     plt.plot(ts, num_spikes,label='neuron {0}'.format(i),)
    #     plt.title("Number of Spikes vs Time Elapsed")
    #     plt.legend()
    #
    #
    #     # if i==0:
    #     rates = np.diff(num_spikes) / data['dt']
    #     plt.figure("isis")
    #     plt.ylabel("Time Between Successive Spikes")
    #     plt.xlabel(r"$Spike Number$")
    #     plt.plot(np.diff(data['O'][i,0:n_spikes]), label='neuron {0}'.format(i))
    #     plt.title("Interspike Intervals vs Spike Number")
    #     plt.legend()

    if show_plots:
        plt.show()

def plot_basic_model_gj(show_plots=True, N=4, k=1, T=10, dt=1e-5):
    name = 'basic_plots'
    this_dir = check_dir(name)

    A = -np.eye(2)
    A[0, 1] = -1
    A[1, 0] = -1

    # A = -.5 * np.eye(2)

    B = np.eye(2)

    D = .1 * gen_decoder(A.shape[0], N, mode='2d cosine')

    theta = 90 * (np.pi / 180)  # in degrees

    k = np.asarray([np.cos(theta), np.sin(theta)])
    k /= 1 * np.linalg.norm(k)

    period = 10
    sin_func = lambda t: np.asarray([np.cos(t * 2 * np.pi / period), np.sin(t * 2 * np.pi / period)])

    # sin_func = lambda t: np.asarray([-1, 1])

    x0 = sin_func(0)

    # x0 = np.asarray([-1, 1])

    lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)
    # pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, lam_v=0)
    net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)



    # pcf_data = pcf_net.run_sim()
    gj_data = gj_net.run_sim()
    data = net.run_sim()
    plot_step = 10

    uA, _ = np.linalg.eig(lds.A)

    # plot error trajectory as scatter, color with time
    _, uA = np.linalg.eig(A)

    ts = data['t'][0:-1:plot_step]
    e =  -uA @ (data['x_hat'][:, 0:-1:plot_step] -   uA.T @ data['x_true'][:, 0:-1:plot_step])
    gj_e =  -(gj_data['x_hat'][:, 0:-1:plot_step] - gj_data['x_true'][:, 0:-1:plot_step])
    gj_ts = gj_data['t'][0:-1:plot_step]




    cmapsc = cm.get_cmap('Reds', lut=len(ts))
    cmapgj = cm.get_cmap('Greens', lut=len(ts))

    mask = np.arange(0, 200)
    plt.show()
    plt.scatter(e[0, mask], e[1, mask], c=ts[mask],marker='s', cmap=cmapsc)
    cbar = plt.colorbar()
    plt.scatter(gj_e[0, mask], gj_e[1, mask], c=gj_ts[mask],marker='.', cmap=cmapgj)
    cbar2 = plt.colorbar()
    plt.title('Error Trajectory in State Space')
    plt.xlabel(r'$\hat{x}_0$')
    plt.ylabel(r'$\hat{x}_1$')

    for j in range(4):
        if j==0:
            plt.scatter(.5*D[0,j],.5*D[1,j], c = 'g', marker= 'x', label=r'$ \frac{d_{j}}{2}$,  $e_{GJ} = \hat{x}-x$')
        else:
            plt.scatter(.5 * D[0, j], .5 * D[1, j], c='g', marker='x')

    _, uA = np.linalg.eig(A)
    uAs = uA
    uA = np.hstack((uA, -uA))

    _, sD, _ = np.linalg.svd(D)

    # Add the patch to the Axes


    plt.gca().add_patch(
     patches.Rectangle((-D[0,0]/2 , -D[1,1]/2 ), D[0,0], D[1,1],linewidth=2, edgecolor='g', facecolor='none',


                          ))
    S = uAs @ (np.hstack((np.diag(sD), np.diag(-sD))) / (2 * np.sqrt(2)))
    sD = np.diag(np.hstack((sD, -sD)))

    sD = uA @ sD
    cos_theta = (sD[0:2,0]).T @ uA[:,1]
    angle = np.arccos(cos_theta) * (180 / (2 * np.pi))

    rect = patches.Rectangle((-sD[0, 0] / 2, -sD[1, 1] / 2), sD[0, 0], sD[1, 1], linewidth=2, edgecolor='r', facecolor='none')
    t=matplotlib.transforms.Affine2D().rotate_deg_around(0, 0, angle)
    rect.set_transform(t + plt.gca().transData)
    plt.gca().add_patch(rect)

    for j in range(4):
        if j==0:

            plt.scatter(S[0, j]  , S[1, j] , c='r', marker='x', label=r'$ U \frac{S_{j}}{2}$,  $e_{SC} = U \hat{y} - x$')
        else:
            plt.scatter( S[0, j] , S[1, j] , c='r', marker='x')



    # for each vector in d, plot the line, uA @    (e.^T d = ||d||^2)
    #  for d in D
    # e0 d0 + e1 d1 = (d0**2 + d1**2) / 2
    # e1 = ( (d0**2 + d1**2) / 2 - e0 d0 ) / d1
    es0 = np.linspace(-np.linalg.norm(D[:, 0] ** 2 / 2), np.linalg.norm(D[:, 0] ** 2 / 2), num=1000)

    bb0 = ((D[0, 0] ** 2 + D[1, 0] ** 2) / 2 - es0 * D[0, 0])  # / D[1,0]
    #plt.plot(es0, bb0)

    S = np.hstack((np.eye(2), - np.eye(2)))

    plt.xlim([-.5, .5])
    plt.ylim([-.5, .5])
    cbar.set_label('Time Elapsed (Dimensionless)')
    plt.legend()
    if show_plots:
        plt.show()

def plot_basic_model(show_plots=True,N = 4, k = 1, T = 10, dt = 1e-5):
    
    name = 'basic_plots'
    this_dir =  check_dir(name)
    

    A = -np.eye(2)
    A[0,1] = -1
    A[1,0] =  1

    #A = -.5 * np.eye(2)

    B = np.eye(2)

    D = .1 * gen_decoder(A.shape[0], N,mode='2d cosine')



    theta  = 90 * (np. pi / 180)  # in degrees

    k = np.asarray([np.cos(theta), np.sin(theta)])
    k /= 1 * np.linalg.norm(k)

    period = 10
    sin_func = lambda t: np.asarray([np.cos(t*2 * np.pi / period), np.sin(t * 2 * np.pi / period )])


    #sin_func = lambda t: np.asarray([-1, 1])

    x0 = sin_func(0)

    #x0 = np.asarray([-1, 1])

    lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)
    #pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, lam_v=0)
    net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)
    gj_net = GapJunctionDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0)



    #pcf_data = pcf_net.run_sim()
    gj_data = gj_net.run_sim()
    data = net.run_sim()
    plot_step = 10

    uA, _ = np.linalg.eig(lds.A)



    #plt.figure()
    #plt.plot(data['t'], data['lds_data']['U'][0,:],label='orig')
    #plt.plot(data['t'], data['U'][0,:], label='rotated')
    #plt.plot(data['t'], (data['Mc'] @data['lds_data']['U'])[0, :], label='input')
    #plt.legend()
    #print(data['sD'])
    #plt.show()
    #assert(False)

    #plot error trajectory as scatter, color with time
    _, uA = np.linalg.eig(A)

    ts = data['t'][0:-1:plot_step]
    e = data['x_hat'][:,0:-1:plot_step] - uA.T @ data['x_true'][:,0:-1:plot_step]
    gj_e = uA.T @ gj_data['x_hat'][:,0:-1:plot_step] - uA.T @ gj_data['x_true'][:,0:-1:plot_step]
    gj_ts = gj_data['t'][0:-1:plot_step]

    cmapsc = cm.get_cmap('Reds', lut=len(ts))
    cmapgj = cm.get_cmap('Greens', lut=len(ts))





    # plt.figure()
    # plt.plot(data['t'],v_est)
    # plt.ylim([-2, 2])
    #v_est = 1 / sD[1] * (data['V'][1, :])
    # plt.plot(data['t'],v_est)

    #plt.plot(data['t'],(( sD[0] * uA[0,0] * data['r'][0,:]) - (sD[0] * uA[0,0] * data['r'][2,:])) / np.sqrt(2))
    #plt.plot(data['t'], data['x_hat'][0,:])

    #plt.plot(data['t'], data['x_true'][0,:] - data['x_hat'][0,:])
    #plt.plot(data['t'], 1 / sD[0] * data['x_hat'][0,:])
    #plt.plot(data['t'], 1 / sD[0] * data['x_true'][0, :])
    mask = np.arange(0,200)
    plt.show()
    plt.scatter(e[0,mask],e[1,mask], c=ts[mask], cmap=cmapsc)
    cbar = plt.colorbar()
    plt.scatter(gj_e[0,mask], gj_e[1, mask],c=gj_ts[mask], cmap=cmapgj)
    cbar2 = plt.colorbar()
    plt.title('Error Trajectory in State Space')
    plt.xlabel(r'$y_0$')
    plt.ylabel(r'$y_1$')

    _, uA = np.linalg.eig(A)
    uAs = uA
    uA = np.hstack((uA, -uA))

    _, sD, _ = np.linalg.svd(D)




    # Add the patch to the Axes
    plt.gca().add_patch(patches.Rectangle((-sD[0]**2/2,-sD[1]**2/2), sD[0]**2, sD[1]**2,  edgecolor='r', facecolor='none',label='+/- $||S_{0/1}||^2 / 2$'))
    # for each vector in d, plot the line, uA @    (e.^T d = ||d||^2)
    #  for d in D
        # e0 d0 + e1 d1 = (d0**2 + d1**2) / 2
        #e1 = ( (d0**2 + d1**2) / 2 - e0 d0 ) / d1
    es0 = np.linspace(-np.linalg.norm(D[:,0]**2/2), np.linalg.norm(D[:,0]**2/2), num = 1000)


    bb0 = ( (D[0,0]**2 + D[1,0]**2) / 2 - es0 * D[0,0] ) #/ D[1,0]
    plt.plot(es0, bb0)


    S = np.hstack((np.eye(2), - np.eye(2)))
    #for j in range(N):
        #plt.scatter(S[0, j], S[1, j], marker='o', label='Neuron {0}'.format(j))

        #j_scale = D[:,j].T @ D[:,j] / 2
        #plt.scatter(D[0,j],  D[1,j], marker='x', label='Neuron {0}'.format(j))
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    cbar2.set_label('Time Elapsed (Dimensionless)')
    plt.legend()

    vmin = -np.min(data['vth'])
    vmax = np.max(data['vth'])

    plt.figure()
    for i in [0, 2]:
        plt.plot(data['t'],data['V'][i,:], label='Neuron %i Voltage'%i)
    plt.legend()
    plt.title("Neuron Membrane Potentials")
    plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$")
    plt.ylabel('Membrane Potential')
    #
    #
    # plt.savefig('plots/basic_plots/membrane_potential_plot.png',bbox_inches='tight')
    #
    # plt.figure()
    # cbar_ticks = np.linspace( start = vmin, stop = vmax,  num = 8, endpoint = True)
    # plt.imshow(data['V'],extent=[0,data['t'][-1], 0,3],vmax=vmax, vmin=vmin,interpolation='none')
    # plt.xlabel(r"Dimensionless Units of $\tau$")
    # plt.axis('auto')
    # cbar = plt.colorbar(ticks=cbar_ticks)
    # cbar.set_label(r'$\frac{v_j}{v_{th}}$')
    # cbar.ax.set_yticklabels(np.round(np.asarray([c / vmax for c in cbar_ticks]), 2))
    # plt.title('Neuron Membrane Potentials')
    # plt.ylabel('Neuron #')
    # plt.yticks([.4,1.15,1.85,2.6], labels=[1, 2, 3, 4])
    #
    _, uA = np.linalg.eig(A)
    uAs = uA
    uA = np.hstack((uA, -uA))

    plt.figure()
    ts = data['t'][0:-1:plot_step]
    plt.plot(ts, (data['x_hat'])[0,0:-1:plot_step],c='r',label='Dimension 0' )
    plt.plot(ts, ( data['x_hat'])[1,0:-1:plot_step],c='g',label='Dimension 1' )
    plt.plot(ts, ( uA.T @ data['x_true'])[0,0:-1:plot_step],c='k',label='True Dynamical System')
    plt.plot(ts, ( uA.T @ data['x_true'])[1, 0:-1:plot_step], c='k')
    plt.title('Network Decode')
    plt.legend()
    plt.ylim([-2, 2])
    plt.xlabel(r'Dimensionless Time $\tau_s$')
    plt.ylabel('Decoded State')


    plt.figure()
    plt.plot(ts, (data['x_hat'])[0,0:-1:plot_step] - (uA.T @ data['x_true'])[0,0:-1:plot_step],c='r',label='Dimension 0' )
    plt.plot(ts, (data['x_hat'])[1,0:-1:plot_step] - (uA.T @ data['x_true'])[1,0:-1:plot_step],c='g',label='Dimension 1' )
    plt.title('Decode Error')
    plt.legend()
    plt.ylim([-2, 2])
    plt.xlabel(r'Dimensionless Time $\tau_s$')
    plt.ylabel('Decode Error')

    # plt.figure()
    # for i in range(4):
    #     plt.plot(data['t'][0:-1:plot_step], data['r'][i,0:-1:plot_step])
    #
    #

    if show_plots:
        plt.show()

def plot_const_driving(show_plots=True,N = 4, T = 10, dt = 1e-5):
      
    def run_plots(T):
        #plot_voltage_prediction()   
        #plot_rate_vs_k()
        #plot_rate_vs_l()
        #plot_xhat_estimate_explicit() 
        plot_per_spike_rmse_vs_phi(T)
        #plot_per_spike_rmse_vs_phi_const_s(T) 
        #plot_per_spike_rmse_vs_phi_const_sk(T)
        #plot_rate_sweep(T)
        #plot_per_spike_rmse_vs_phi_const_k(T) 
        #plot_rmse_sweep()
        #plot_lambda_0()
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

        def nrmse(phi):
            t1 = (1 + np.exp(-1 / phi)) / (1 - np.exp(-1 / phi))
            return np.sqrt(.25 * t1 ** 2 - phi * .5 * t1)
            
            

        
        B = np.eye(2)
        x0 = np.asarray([1, 0])
        

        d=2
        
        
        print('\t\tPlotting Rate vs Drive Strength k\n')
        ks = np.logspace(-5, -.0001, num = 20)
        ks = np.hstack((ks, np.logspace(0,5,num=20)))

        rates = np.zeros(ks.shape)
        ss = np.zeros(ks.shape)
        lams = -np.asarray([10, 1, .1])
        plt.figure()

        for l in lams:
            for i, k in enumerate(ks):
                A = l * np.eye(2)

                sin_func = lambda t: k * np.asarray([1, 0])
                lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)

                print('{0} / {1}'.format(i + 1, len(ks)))
                D =   1 * np.hstack((
                                np.eye(d),
                                np.zeros((d, N - d))
                                ))
                net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
                data = net.run_sim()

                midtime = data['t'][int(len(data['t']) * (3/4))]
                rates[i] = np.sum(data['O'][0, :] >= midtime) / (T - midtime)
                #spikes = data['O'][0,:data['spike_nums'][0]]
                #isis = np.diff(spikes)
                #misi = np.mean(isis)
                #rates[i] = 1/misi
                ss[i] = 1
                #rmses[i] = per_spike_rmse_numerical(data, 0)[0] / k

            # plt.figure()
            # plt.loglog(rates, nrmse_phi(rates), label='Derived Expression',linewidth=4)
            # plt.loglog(rates, rmses, 'x', label='Numerical Simulation')
            # plt.ylabel(r'NRMSE$')
            # plt.xlabel(r'Neuron Firing Rate $\phi$')
            # plt.title('Normalized per-Spike RMSE vs Firing Rate')
            # plt.legend()
            # plt.savefig(this_dir + '/' + 'ps_rmse_const_driving.png', bbox_inches='tight')
            #

            plt.loglog(ks, sc_phi(ks, 1, l), label='Derived Expression, $\Lambda = {0}$'.format(l),linewidth=4)
            plt.loglog(ks, rates, 'x')
        plt.xlabel(r'Drive Strength Ratio $\frac{(\beta k)_j}{\sigma_j}$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate Response to Constant Driving')
        plt.legend()
        plt.axhline(y=1/dt, ls='--')
        plt.savefig(this_dir + '/' + 'phi_const_driving.png', bbox_inches='tight')
        
    def plot_rate_vs_l_sweep_s():
        
        B = np.eye(2)
        x0 = np.asarray([.5, 0])
        d=2
        s = 1
        
        sin_func = lambda t :  np.asarray([1, 1])
        
    
        lams = -np.logspace(-1, -.5, num=10)
        
        print('\t\tPlotting Rate vs Drive Strength k\n')
        rates =np.zeros(lams.shape)
        ss = np.logspace(-1, 0 , num = 5)
        rmses=np.zeros(lams.shape)
        plt.figure()
        count=1
        for s in ss:
            for i, l in enumerate(lams):
                print('{0} / {1}'.format(count, len(lams)*len(ss)))
                
                D =  s * np.hstack((
                            np.eye(d),
                            np.zeros((d, N - d))
                            ))
                A =  - l * np.eye(2)
    
                lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
    
                net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
                data = net.run_sim()
                
             
                rates[i] = data['spike_nums'][0]   /  data['t'][-1]  
                rmses[i] = per_spike_rmse_numerical(data, 0)[0] 
                count+=1
            plt.loglog(np.abs(lams), rates,'x-', label='$\sigma_j={0}$'.format(s))
        plt.xlabel(r'$|\Lambda_j|$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate vs $\Lambda_j$')
        plt.legend()
        plt.savefig(this_dir + '/' + 'firing_rate_const_driving_s_sweep.png', bbox_inches='tight')
        
    def plot_rate_vs_l_sweep_k():
        
        B = np.eye(2)
        x0 = np.asarray([.5, 0])
        d=2
        s = 1
        D =  s * np.hstack((
            np.eye(d),
            np.zeros((d, N - d))
            ))
        
    
        lams = -np.logspace(-1, -.5, num=10)
        
        print('\t\tPlotting Rate vs Drive Strength k\n')
        rates =np.zeros(lams.shape)
        rmses=np.zeros(lams.shape)
        
        ks = np.logspace(-1, 0, num=5)
        plt.figure()
        count=1
        for k in ks:
            for i, l in enumerate(lams):
                print('{0} / {1}'.format(count, len(lams)*len(ks)))
                sin_func = lambda t : k * np.asarray([1, 1])


                A =  - l * np.eye(2)
    
                lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
    
                net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
                data = net.run_sim()
                
             
                rates[i] = data['spike_nums'][0]   /  data['t'][-1]  
                count+=1
            plt.loglog(np.abs(lams), rates,'x-', label='$(\beta k)_j={0}$'.format(k))
        plt.xlabel(r'$|\Lambda_j|$')
        plt.ylabel(r'Neuron Firing Rate $\phi$')
        plt.title('Neuron Firing Rate vs $\Lambda_j$')
        plt.legend()
        plt.savefig(this_dir + '/' + 'firing_rate_const_driving_k_sweep.png', bbox_inches='tight')
        
    def plot_xhat_estimate_explicit():
        print('\t\tPlotting Network Estimate Comparison to Explicit Equation\n')
        
        def y_hat_explicit(t, phis, s, xi_0s):  
            
            
            y_hats = np.zeros((len(phis), len(t)))
            
            for j in range(len(phis)):
                eq_pred_height = s[j] / (1 - np.exp(-1/phis[j])) 
                y_hats[j,:] = np.exp(- np.mod(t - xi_0s[j], 1/phis[j])) * eq_pred_height
            
            return y_hats
    
     
            

        A = -np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([.5, .5])
        d = A.shape[0]
        sigma = .5
        
        D =  sigma * np.eye(N)[:d,:]
        T = 10
        s = np.ones((N,)) * sigma
        
        k = np.asarray([1, 1])
        sin_func = lambda t :  k 
        
        lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
        net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
        data = net.run_sim()
        
        
        rates = data['spike_nums'] / data['t'][-1]
        rates = phi_bk(sigma, 1) * np.ones((N,))
        idx = 0  
        plt.figure()
        plt.plot(data['t'], y_hat_explicit(data['t'], rates, s, data['O'][:,0])[0,:],'--', label='Derived Expression',linewidth=2)
        plt.plot(data['t'],data['x_hat'][idx,:],'r',label='Simulated Network Estimate',linewidth=2,alpha=.5)
        plt.plot(data['t'],data['x_true'][idx,:],'k', label = 'Target System', linewidth=2)                
        plt.xlabel(r"Simulation Time (Dimensionless Units of $\tau$)")
        plt.ylabel('Network Decode Dimension 0')
        plt.legend()
        plt.title('Predicted Network Decode Comparison')
        plt.savefig(this_dir + '/' + 'network_decode_long_term_estimate.png', bbox_inches='tight')

    def plot_rate_sweep(T):
        def sweep_ls_ss_k_const(T):
            # keep l constant, sweep ks vs ss
            bks = np.logspace(-3, 3, num=1000)
            ss = np.logspace(-3, 3, num=1000)
            ls = -np.logspace(-3, 3, num=1000)

            res = 30

            ss_num = [.001, .005, .01, .1]
            ls_num = -np.logspace(-3, 0, num=res)
            # run sims
            print("Fixing L sweeping bks and ss")
            #theoretical plot
            # # keep ks constant, sweep ls vs ss
            plt.figure()
            ls_grid, ss_grid = np.meshgrid(ls, ss)
            Z = sc_phi(1 / (ss_grid * np.abs(ls_grid)) )
            plt.pcolormesh(np.abs(ls_grid), ss_grid, Z, cmap='jet', norm=LogNorm())
            plt.plot(np.abs(ls), 2 / np.abs(ls) , '--', label=r'$\frac{(\beta k)_j}{|\Lambda_j|} = \frac{\sigma}{2}$')
            cbar = plt.colorbar()
            plt.xscale('log')
            plt.yscale('log')
            cbar.set_label('$\phi_j$')
            plt.xlabel('$\|\Lambda_j|$')
            plt.ylabel(r'$\sigma_j$')
            plt.title(r'Sweep $\Lambda_j$, $\sigma_j$, $(\beta k)_j = 1$')

            B = np.eye(2)
            x0 = np.asarray([1, 0])
            k = 1 * np.asarray([1, 1])
            d = 2
            rates = np.empty((res, res))
            count = 1

            plt.figure()
            rates_pred = np.zeros((len(ss_num), len(ls_num)))

            for i, s in enumerate(ss_num):

                for j, l in enumerate(ls_num):
                    print('{0} / {1}'.format(count, len(ss_num) * len(bks_num)))

                    rates_pred[i, j] = sc_phi(1, s, l)

                    D = s * np.hstack((
                        np.eye(d),
                        np.zeros((d, N - d))
                    ))
                    A = -np.eye(2)

                    lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)
                    net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob=1)
                    data = net.run_sim()
                    # spikes = data['O'][0, :data['spike_nums'][0]]
                    # isis = np.diff(spikes)
                    # misi = np.mean(isis)
                    midtime = data['t'][int(len(data['t']) // 2)]
                    rates[i, j] = data['spike_nums'][0] / data['t'][
                        -1]  # np.sum(data['O'][0, :] >= midtime) / (T - midtime)

                    count += 1

                plt.plot(bks_num, rates_pred[i, :], label='$\sigma_j = ${0}'.format(s))
                plt.plot(bks_num, rates[i, :], 'x')

            plt.xscale('log')
            plt.xlabel(r"$(\beta k)_j$")
            plt.ylabel(r'$\phi_j$')
            plt.title('Firing Rate with $|\Lambda_j| = 1$')
            plt.legend()

        def sweep_ks_ss_l_const(T):
            # keep l constant, sweep ks vs ss
            bks = np.logspace(-2, 0, num=1000)
            ss = np.logspace(-2, 0, num=1000)
            ls = -np.logspace(-2, 0, num=1000)

            res = 30
            bks_num = np.logspace(-2,0, num=res)
            ss_num = [.001, .005, .01, .1]
            ls_num = -np.logspace(-2, 0, num=res)
            # run sims
            print("Fixing L sweeping bks and ss")

            # theoretical plot
            ss_grid, bks_grid = np.meshgrid(ss, bks)
            plt.figure()
            ss_grid_num, bks_grid_num = np.meshgrid(ss_num, bks_num)
            Z = sc_phi(bks_grid, ss_grid, 1)
            plt.plot(ss, ss / 2, '--', label=r'$\frac{(\beta k)_j}{|\Lambda_j|} = \frac{\sigma}{2}$')
            plt.pcolormesh(ss_grid, bks_grid, Z, cmap='jet', norm=LogNorm(vmax=2/dt))
            plt.xlabel('$\sigma_j$')
            cbar = plt.colorbar()
            plt.ylabel(r'$(\beta k)_j$')
            plt.xscale('log')
            plt.yscale('log')
            cbar.set_label('$\phi_j$')
            plt.xlabel('$\sigma_j$')
            plt.ylabel(r'$\frac{(\beta k)_j}{|\Lambda_j|}$')
            plt.title(r'Sweep $\sigma_j$, $(\beta k)_j$, $\Lambda_j = -1$')

            A = -np.eye(2)
            B = np.eye(2)
            x0 = np.asarray([1, 0])
            d = A.shape[0]
            T = T
            rates = np.empty((res, res))
            count = 1
            bks_num = np.logspace(-3, np.log(.5), num=res)

            plt.figure()
            rates_pred = np.zeros((len(ss_num), len(bks_num)))



            for i, s in enumerate(ss_num):

                for j, k_scale in enumerate(bks_num):
                    print('{0} / {1}'.format(count, len(ss_num)*len(bks_num)))

                    rates_pred[i,j] = sc_phi(k_scale, s, -1)
                    #rates[rates > 1 / dt] = np.nan
                    #rates_pred[rates_pred > 1 / dt] = np.nan

                    # if np.isnan(rates_pred[i,j]):
                    #     rates[i,j] = np.nan
                    #     continue

                    # if s > k_scale / 2:
                    #     count += 1
                    #     continue

                    k = k_scale * np.asarray([1, 1])
                    sin_func = lambda t: k
                    D = s * np.hstack((
                        np.eye(d),
                        np.zeros((d, N - d))
                    ))

                    lds = sat.LinearDynamicalSystem(x0, A, B, u=sin_func, T=T, dt=dt)
                    net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob=1)
                    data = net.run_sim()
                    #spikes = data['O'][0, :data['spike_nums'][0]]
                    #isis = np.diff(spikes)
                    #misi = np.mean(isis)
                    midtime = data['t'][int(len(data['t'])//2)]
                    rates[i,j] = data['spike_nums'][0] / data['t'][-1]#np.sum(data['O'][0, :] >= midtime) / (T - midtime)

                    count += 1

                plt.plot(bks_num, rates_pred[i,:], label='$\sigma_j = ${0}'.format(s))
                plt.plot(bks_num, rates[i,:], 'x')

            plt.xscale('log')
            plt.xlabel(r"$(\beta k)_j$")
            plt.ylabel(r'$\phi_j$')
            plt.title('Firing Rate with $|\Lambda_j| = 1$')
            plt.legend()

        # for a few values of lambda, sweep s vs bk numerically

        # keep ss constant sweep bks ls
        # plt.figure()
        # ls_grid, bks_grid = np.meshgrid(ls, bks)
        # Z = sc_phi(bks_grid / np.abs(ls_grid))
        # plt.pcolormesh(np.abs(ls_grid), np.abs(bks_grid), Z, cmap='jet', norm=LogNorm())
        # plt.plot(np.abs(ls),  np.abs(ls) / 2, '--', label=r'$\frac{(\beta k)_j}{|\Lambda_j|} = \frac{\sigma}{2}$')
        # cbar = plt.colorbar()
        # plt.xscale('log')
        # plt.yscale('log')
        # cbar.set_label('$\phi_j$')
        # plt.xlabel('$\|\Lambda_j|$')
        # plt.ylabel(r'$(\beta k)_j$')
        # plt.title(r'Sweep $(\beta k)_j$, $\Lambda_j$, $\sigma_j,= 1$')
        #
        # # keep ks constant, sweep ls vs ss
        # plt.figure()
        # ls_grid, ss_grid = np.meshgrid(ls, ss)
        # Z = sc_phi(1 / (ss_grid * np.abs(ls_grid)) )
        # plt.pcolormesh(np.abs(ls_grid), ss_grid, Z, cmap='jet', norm=LogNorm())
        # plt.plot(np.abs(ls), 2 / np.abs(ls) , '--', label=r'$\frac{(\beta k)_j}{|\Lambda_j|} = \frac{\sigma}{2}$')
        # cbar = plt.colorbar()
        # plt.xscale('log')
        # plt.yscale('log')
        # cbar.set_label('$\phi_j$')
        # plt.xlabel('$\|\Lambda_j|$')
        # plt.ylabel(r'$\sigma_j$')
        # plt.title(r'Sweep $\Lambda_j$, $\sigma_j$, $(\beta k)_j = 1$')

        sweep_ks_ss_l_const(T)
    def plot_rmse_sweep():
        def per_spike_rmse(bkl, s, phi):
            return np.sqrt(
                bkl**2 + 2*phi*bkl*s + .5*phi*(s**2) * (1 + np.exp(-1/phi)) / (1 - np.exp(-1/phi))
                )

        
        '''
        Sweep two parameters, (Bk)j / lj and sigmaj, 
        compute the per-spike rmse numericaly and via known equation
        plot the results in a 2d scatter plot. 
        '''
        A = -np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([.5, .5])
        d = A.shape[0]
        T = 10
        res = 10
        kres = res
        sres = res
        
#        ss = np.logspace(-2,0, num=sres)
        
        rmses = np.zeros((kres,))
        rates = np.zeros((kres,))
        rmses_pred = np.zeros(rmses.shape)
        
        s = .01
        
        # k >= s^2/2
        
        ks = np.logspace(np.log10(s**2/2), 2, num=kres)
        for i, k_scale in enumerate(ks):
            k = k_scale * np.asarray([1, 1])
            sin_func = lambda t :  k 
            D = s * np.hstack((
                        np.eye(d),
                        np.zeros((d, N - d))
                        ))
                
            lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
            net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
            data = net.run_sim()
            rates[i] = phi_bk(s, k_scale)
            try:
                rmses[i] = per_spike_rmse_numerical(data, 0)[0]
            except Exception as e:
                print("warning: ",e)
                continue
    
            
            rmses[i] = per_spike_rmse_numerical(data, 0)[0]
            rmses_pred[i] = per_spike_rmse(-k_scale, s, rates[i])
        plt.figure()
        plt.loglog(ks,rmses,'x')
        plt.loglog(ks,rmses_pred)
        plt.show()

        
#         for i, k_scale in enumerate(ks):
#             for j, s in enumerate(ss):
#                 if k_scale < s/2:
#                     continue
#                 else:
#                     k = k_scale * np.asarray([1, 1])
#                     sin_func = lambda t :  k 
#                 
#                     D = s * np.hstack((
#                         np.eye(d),
#                         np.zeros((d, N - d))
#                         ))
#                 
#                     lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
#                     net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
#                     data = net.run_sim()              
#                     try:
#                         rmses[i,j] = per_spike_rmse_numerical(data, 0)[0]
#                     except Exception as e:
#                         print("warning: ",e)
#                         continue
#                     rates[i,j] = data['spike_nums'][0] / data['t'][-1]
#                     rmses_pred[i,j] = per_spike_rmse(k_scale, s, rates[i,j])
# 
# 
#         ks_m, ss_m = np.meshgrid(ks,ss)
#         
#         
#         from matplotlib import colors 
#         
#         idx = 0
#         
# 
#         
#         plt.figure()
#         #plt.imshow(rmses_pred, origin='lower',norm=LogNorm())
#         plt.scatter(x=ks_m, y=ss_m, c=rmses, marker='x',s=200, )
#         plt.scatter(x=ks_m, y=ss_m, c=rmses_pred)
#         plt.plot(ks, ss/2 ,'--',label=r'$\frac{(\beta k)_j}{|\Lambda_j|}= \frac{\sigma_j}{2}$')
#         ax = plt.gca()
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         plt.legend(bbox_to_anchor=(1.33, 1.13), ncol=1)
#         plt.xlabel(r'$\frac{(\beta k)_j}{|\Lambda_j|}$')
#         plt.ylabel(r'$\sigma_j$')
#         cbar = plt.colorbar()
#         cbar.set_label("per-Spike RMSE")
#         plt.title('Network Estimate RMSE per Spike')
#         #plt.savefig(this_dir + '/' + 'rmse_sp_const_driving.png', bbox_inches='tight') 
#         
#         rmses_masked = np.zeros((rmses.shape))
#         
#         fig = plt.figure()
#         ax = fig.gca(projection='3d')
#         rmses_masked[~np.isnan(rmses)] =rmses[~np.isnan(rmses)] 
#         
#         rmses_masked[np.isnan(np.log(rmses_masked))] = np.exp(5)
#         rmses_masked[np.isinf(np.log(rmses_masked))] = np.exp(5)
        
#        # Plot the surface.
#         ax.plot_surface(ks_m, ss_m ,np.log(rmses_masked),zorder=1, cmap=cm.get_cmap('jet'))
#         ax.plot3D(ks, ss/2, np.max(np.log(rmses_masked))*np.ones(ks.shape),'--',zorder=5)
#         #ax = plt.gca()
#         #ax.set_yscale('log')
#         #ax.set_xscale('log')
#         #plt.legend(bbox_to_anchor=(1.33, 1.13), ncol=1)
#         plt.xlabel(r'$\frac{(\beta k)_j}{|\Lambda_j|}$')
#         plt.ylabel(r'$\sigma_j$')
#         #cbar = plt.colorbar()
#         #cbar.set_label("Log per-Spike RMSE")
#         plt.title('Network Estimate RMSE per Spike')        
  


    def plot_per_spike_rmse_vs_phi(T):
        def nrmse(phi):
            return np.sqrt(1 - 2*phi * np.tanh(1 / (2 * phi)) )


        print('\t\tPlotting  RMSE vs phi')
        SCALE = 1
        ss = np.logspace(-4,4,num=50)
        d=2
        k_scale = 1

        l = -1*SCALE
        A = l*np.eye(2)
        B = np.eye(2)*SCALE

        rmses = np.zeros(ss.shape)
        rmse_stds = np.zeros(rmses.shape)
        rmse_preds = np.zeros(rmses.shape)
        rates = np.zeros(rmses.shape)
        count = 1
        plt.figure()



        for i, s in enumerate(ss):
            print('{0} / {1}'.format(count, len(ss)))
            x0 = k_scale * np.asarray([1, 1])
            k = k_scale * np.asarray([1, 1])
            rate_pred = sc_phi(k_scale, s, np.abs(l))
            num_cycles = 100
            T = num_cycles / rate_pred

            D = s * np.hstack((
                np.eye(d),
                np.zeros((d, N - d))
            ))

            if T <= dt * num_cycles:
                T = 10

            if np.isnan(T) or np.isinf(T):
                T = 100
            print('T = ',T)

            lds = sat.LinearDynamicalSystem(x0, A, B, u=lambda t: k, T=T, dt=dt)
            net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob=1)

            if T >= 50:
                lds = sat.LinearDynamicalSystem(x0, A, B, u=lambda t: k, T=T, dt=1e-3)
                net = SelfCoupledNet(T=T, dt=1e-3, N=N, D=D, lds=lds, t0=0, spike_trans_prob=1)


            data = net.run_sim()

            # discard data before first spike
            begin = data['O'][0,0]

            mask  = data['t'] >= begin
            data['t'] = data['t'][mask]
            data['error'] = data['error'][:,mask]
            data['x_hat'] = data['x_hat'][:,mask]
            data['x_true'] = data['x_true'][:,mask]

            #
            # print(k_scale)
            # plt.figure()
            # plt.plot(data['t'],data['x_hat'][0,:])
            # plt.xlim([0, 100])
            # plt.plot(data['t'],data['x_true'][0,:])
            # plt.show()

            rates[i] = (data['spike_nums'][0]) / (data['t'][-1] - data['t'][0])

            rmse_preds[i] = nrmse(rates[i])
            mean, std = per_spike_rmse_numerical(data, 0)
            rmses[i] = mean / k_scale
            rmse_stds[i] = std
            count += 1

        #phis_cont = np.logspace(-2,5,num=100)
        plt.plot(rates, rmse_preds,  label='Derived Expression')
        plt.plot(rates, rmses, 'x',label='Numerical Simulation')
       # plt.plot(phis_cont, nrmse(phis_cont), label='Derived Expression')
        plt.xscale('log')
        plt.yscale('log')
        #plt.errorbar(rates, rmses,yerr=rmse_stds)
        plt.xlabel(r'$\phi_j$')
        plt.ylabel(r'NRMSE')
        plt.legend()
        #plt.plot(rates,np.sqrt(.25 - rates/2),'--')
        plt.title(r'Network Estimate RMSE vs Firing Rate')
        plt.savefig(this_dir + '/' + 'rmse_sp_vs_phi_const_driving.png', bbox_inches='tight')
        
    def plot_per_spike_rmse_vs_phi_const_sk(T): 
        print('\t\tPlotting Per Spike RMSE vs k and phi')    
        

        def per_spike_rmse(bkl, s, phi): 
            result = np.sqrt(
                bkl**2 
                + 
                2*phi*bkl*s 
                + 
                .5*phi*(s**2) * 
                (1 + np.exp(-1/phi)) / (1 - np.exp(-1/phi))
                )
            if np.isnan(result):
                return -1
            else:
                return result
        
        def approx(phi,s):
            p = phi
            #return np.sqrt(s**2/2 * p / np.tanh(1/(2*p)))
            return np.sqrt(1 + 2*p*s)
            
        
        lams = -np.linspace(.1, 2,num=10)
        # 0 
        # bk / l > s / 2
        # 2 bk / l > s
        # 2 bk / s l > 1
        # 2 bk / s > l

        k = np.asarray([1, 1])
        rmses = np.zeros(lams.shape)
        rmse_stds = np.zeros(lams.shape)
        rmse_preds = np.zeros(rmses.shape)
        rmse_apps = np.zeros(rmses.shape)
        rates = np.zeros(lams.shape)
        s = 1
        d = 2
        D = s * np.hstack((
                            np.eye(d),
                            np.zeros((d, N - d))
                            ))
        bkls = np.zeros(rates.shape)
        count = 1
        sin_func = lambda t : k 

        
        for j,l in enumerate(lams):
      
            print('{0} / {1}'.format(count, len(lams)))
            
            
            A =   l * np.eye(2)
            B = np.eye(2)
            x0 = np.asarray([.5, 0])        
            bkl = 1 / l
            bkls[j] = np.abs(bkl)
            lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
            net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
            data = net.run_sim()

            rates[j] = data['spike_nums'][0] / data['t'][-1]
            rmse_preds[j] = per_spike_rmse(bkl, 1, rates[j])
            #rmse_apps[j] = approx(rates[j],s)
            (mean, std) = per_spike_rmse_numerical(data, 0)
            rmses[j] = mean 
            rmse_stds[j] = std       
            count += 1
        
            
        
    
        plt.figure()
        plt.plot(np.abs(lams), rmse_preds, linewidth=4, label = 'Derived Expression')
        plt.plot(np.abs(lams), rmses,'x',label='Numerical Simulation',linewidth=4,markersize=10)
        #plt.plot(np.abs(lams), rmse_apps,'--', label='Approximate Expression',linewidth=4,markersize=10)
        #plt.errorbar(np.abs(lams), rmses, yerr = rmse_stds, fmt='none')


               
        plt.xlabel(r'$|\Lambda_j| / \frac{(\beta k)_j}{\sigma_j}$')
        plt.ylabel(r'per-Spike RMSE')
        plt.legend()
        plt.title(r'Network Estimate RMSE per Spike vs $\Lambda_j$')
        #plt.savefig(this_dir + '/' + 'rmse_sp_vs_phi_const_driving.png', bbox_inches='tight')    
    
    def plot_per_spike_rmse_vs_phi_const_k(T): 
        print('\t\tPlotting Per Spike RMSE vs k and phi')    
        

        def per_spike_rmse(bkl, s, phi): 
            result = np.sqrt(
                bkl**2 
                +
                2*phi*bkl*s 
                + 
                .5*phi*(s**2) * 
                (1 + np.exp(-1/phi)) / (1 - np.exp(-1/phi))
                )
            if np.isnan(result):
                return -1
            else:
                return result
        
        
        
        
        
      
        k = np.asarray([1, 1])
        ss = np.linspace(.1, 1, num=3)
        rmses = np.zeros(ss.shape)
        rmse_stds = np.zeros(ss.shape)
        rmse_preds = np.zeros(rmses.shape)
        rates = np.zeros(ss.shape)
        d = 2
        lams = -np.asarray([.5, 1, 2])
        bkls = np.zeros(rates.shape)
        count = 1
        for j,l in enumerate(lams):
            for i, s in enumerate(ss):
                print('{0} / {1}'.format(count, len(ss)*len(lams)))
                
                sin_func = lambda t : k 
                D = s * np.hstack((
                            np.eye(d),
                            np.zeros((d, N - d))
                            ))
                
           
                
                A =   l * np.eye(2)
                B = np.eye(2)
                x0 = np.asarray([.5, 0])        
                #T = T / np.abs(l)
                bkl = 1 / l
                bkls[i] = np.abs(bkl)
                lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
                net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
                data = net.run_sim()
 
                rates[i] = data['spike_nums'][0] / data['t'][-1]
                rmse_preds[i] = per_spike_rmse(bkl, s, rates[i])
    
                (mean, std) = per_spike_rmse_numerical(data, 0)
                rmses[i] = mean 
                rmse_stds[i] = std       
                count += 1
            
            if j==0:
                plt.figure()
                plt.semilogy(ss, rmse_preds, linewidth=4, label = r'$\Lambda_j={0}$'.format(l))
                plt.semilogy(ss, rmses, 'x', c='g',label='Numerical Simulation',linewidth=4,markersize=10)
                
            else:
                plt.semilogy(ss, rmse_preds, linewidth=4, label = r'$\Lambda_j={0}$'.format(l))
                plt.semilogy(ss, rmses, 'x',c='g',linewidth=4,markersize=10)
                
                #1 / abs(l) > s/2
                #abs(l) < 2 /s
                # s < 2 / abs(l)
                
            #plt.axvline(x=2 / np.abs(l), ls='--',c='k')
          
        plt.xlabel(r'$\sigma_j$')
        plt.ylabel(r'per-Spike RMSE')
        plt.legend()
        plt.title(r'Network Estimate RMSE per Spike vs $\frac{(\beta k)_j}{\Lambda_j}$')
        plt.savefig(this_dir + '/' + 'rmse_sp_vs_phi_const_driving.png', bbox_inches='tight')    

    def plot_lambda_0():
        ks = np.logspace(-1,1,num=1000)
        ss = np.logspace(-1,1,num=1000)
        extent=(ss[0],ss[-1],ks[0],ks[-1])
        y0= 1
        
      
        ss, ks = np.meshgrid(ss, ks)
        Z = phi_lam0(ks, y0, ss)
        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        # Plot the surface.
        #surf = ax.plot_surface(ss,ks, Z, cmap='gray',
        #                       linewidth=0, antialiased=False)
                
        rmax = np.max(Z)
        rmin = np.min(Z)
        plt.title(r'Firing Rate as $|\Lambda_j \to 0|$')
        plt.imshow(Z, cmap='jet',origin='lower',norm=LogNorm(),extent=extent,vmax=rmax, vmin=rmin)
        plt.xlabel('$\sigma_j$')
        cbar = plt.colorbar()
        plt.ylabel(r'$(\beta k)_j$')
        cbar.set_label('$\phi_j$')
        

        
        # now sweep through discrete set for particular lambda and plot image
        res = 20
        ks = np.logspace(-1,1,num=res)
        ss = np.logspace(-1,1,num=res)
        
        B = np.eye(2)
        x0 = np.asarray([1, 0])

        d=2
        rates =np.zeros((res,res))
        
        count = 1
        
        for i, k in enumerate(ks):
            for j, s in enumerate(ss):
                print('{0} / {1}'.format(count, len(ks)*len(ss)))
                A =  - 1e-16*np.eye(2)
                sin_func = lambda t : k * np.asarray([1, 0])
                lds = sat.LinearDynamicalSystem(x0, A, B, u = sin_func , T = T, dt = dt)
                
                D =  s*np.hstack((
                                np.eye(d),
                                np.zeros((d, N - d))
                                ))
                net = SelfCoupledNet(T=T, dt=dt, N=N, D=D, lds=lds, t0=0, spike_trans_prob = 1)
                data = net.run_sim()
            
         
                rates[i,j] = data['spike_nums'][0]   /  data['t'][-1]   
                count += 1
        
        ss, ks = np.meshgrid(ss, ks)
                
        fig = plt.figure()
                
        plt.title(r'Firing Rate as $|\Lambda_j \to 0|$')
        plt.imshow(rates, cmap='jet',origin='lower',norm=LogNorm(),extent=extent,vmax=rmax, vmin=rmin)
        plt.xlabel('$\sigma_j$')
        cbar = plt.colorbar()
        plt.ylabel(r'$(\beta k)_j$')
        cbar.set_label('$\phi_j$')
                
    name = 'const_drive_strength'
    this_dir =  check_dir(name)
    
    print('\tPlotting Constant Driving Strength\n')
    
    run_plots(T)

def plot_pcf_gj_sc_comparison(show_plots=True,N = 32, T = 1000, dt = 1e-3, num_sim_points = 10):   
     
    def run_plots(num_sim_points = 10):   
        #plot_demos() 
        #plot_pcf_gj_sc_long_term_estimates_explicit() 
        #plot_pcf_gj_sc_rates(num_sim_points) 
        #plot_pcf_gj_sc_constant_stim_decode()
        #plot_pcf_gj_membrane_trajectories()
        plot_pcf_gj_sc_per_spike_rmse(T)
        
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
               # (sc_data, 'Self-Coupled',lambda t : x_hat_explicit_s(t, sD[0]), 'x', 'r', 20) ,
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
    
    def plot_pcf_gj_sc_per_spike_rmse(T):
        def rmse_per_spike_pcf(phi):
            return np.sqrt(1 - 2 * phi * np.tanh(1/ (2 * phi)))


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
#                   d
        
        print('\t\tPlotting Rate Versus per-spike RMSE\n')
        A =  - np.eye(2)
        B = np.eye(2)
        x0 = np.asarray([1, 0])
        
        ks = np.logspace(-5, 1,num=30 )
       
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
            pcf_net = ClassicDeneveNet(T=T, dt=dt, N=N, D=D, lds=lds, lam_v=0)
            
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
            (pcf_data, 'PCF', '.', 'b', 20)
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

            plt.loglog(rate_measurements[model_name], rmses_numerical[model_name], marker=marker, c=color, markersize= msize, label=model_name)

        rates_cont = np.logspace(-2, 4,num=1000)
        plt.loglog(rates_cont, rmse_per_spike_pcf(rates_cont), '--',label='Derived Expression')

        plt.ylabel(r' NRMSE')
        plt.xlabel(r'Neuron Firing Rate $\phi$')
        plt.title('NRMSE vs Neuron Firing Rate')
        plt.legend()
        plt.savefig(this_dir + '/' + 'sc_pcf_gj_nrmse_vs_phi.png', bbox_inches='tight')
        
    name = 'pcf_gj_sc_comparison'
    this_dir =  check_dir(name)
    
    
    run_plots(num_sim_points)
    
    
    #plot_pcf_gj_sc_per_spike_rmse(20) 
    if show_plots:
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

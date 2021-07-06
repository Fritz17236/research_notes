import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import queue

def W(t, tau_pos = 10, tau_neg = 40, tau_R = 4, beta = 1.4, eta = 1e-7):
    '''
    Spike-Timing-Dependent Plasticity (STDP) Weighting Function
    Computes the weight change for a synapse given the time lag between spikes of the
    pre-synaptic and post-synaptic neurons.

    :param t: (np array) time lag between pre and postynaptic spikes (ms)
    :param tau_pos: long term potentition (LTP) decay time (ms)
    :param tau_neg: long term depression (LTD) decay time (ms)
    :param tau_R: decay assymetry paramter (ms)
    :param beta: ltp/ltd ratio
    :param eta: per-spike weight update / learning rate
    :return: The total change in weight value
    '''
    dW = np.zeros(t.shape)
    dW[t > 0] = np.exp(-np.abs(t[t > 0]) / tau_pos)
    dW[t < 0] =  -(beta / tau_R) * np.exp(-(np.abs(t[t < 0])) / tau_neg)

    return eta * dW


# need synaptic delay
# use chalk paper to update r and o



## Network Architecture
noise_scale = .001
mu = 0*10**-6
nu = 0*10**-5
lam_v = 1
lam_d = 1

# Define LDS
A = np.zeros((2, 2))
A[0, 1] = -1
A[1, 0] = 1
#A = -np.eye(2)
x0 = np.asarray([10, 0])

# Decoder Matrix
N = 32
D = np.zeros((2, N))
np.random.seed(0)
D = np.random.randn(2, N)
for j in range(N):
    theta = 2 * np.pi / N * (j + 1)
    D[:, j] /= np.linalg.norm(D[:, j])
D *= .1

# Define Voltage Dynamics Matrices
Mv = - lam_v * np.eye(N)
Mr = D.T @ (A + lam_d * np.eye(A.shape[0])) @ D
Mo = - (D.T @ D + mu * lam_d**2 * np.eye(N))

#Voltage Thresholds
v_th = (np.diag(D.T @ D) + nu * lam_d + mu * lam_v) / 2

# Initialize simulation
dt = 1e-4
T = 10
ts = np.arange(start=0, stop=T, step=dt)
num_timesteps = len(ts)

V = np.zeros((N, num_timesteps))
r = np.zeros(V.shape)
spikes = {i : [] for i in range(N)}
last_spikes = np.zeros((N,1))
x_true = np.zeros((2, num_timesteps))
x_true[:,0] = x0

r[:,0] = nnls(D, x0)[0]
Mrs = np.zeros((N**2, num_timesteps))
Mrs[:, 0] = Mr.flatten()
noise = np.random.rand(N, num_timesteps) * noise_scale

t_delay = .001  # axonal delay
num_timesteps_delay = int(t_delay / dt)

print("actual delay: {0}".format(dt * num_timesteps_delay))

spike_queue = queue.Queue(maxsize=num_timesteps_delay)
for j in range(num_timesteps_delay - 1):
    spike_queue.put(None)

#simulate voltage & r dynamics
for t_idx, t in enumerate(ts[:-1]):
        dV = Mv @ V[:, t_idx] + Mr @ r[:, t_idx] #+ D.T @ [1, 0]
        V[:, t_idx + 1] = V[:,t_idx] + dt * dV + noise[:, t_idx]

        dr = - lam_d * r[:,t_idx]
        r[:,t_idx + 1] = r[:,t_idx] + dt * dr

        # check for spikes
        if np.any(V[:, t_idx + 1] > v_th):
            spike_idx = np.argmax(V[:, t_idx + 1])
            spikes[spike_idx].append(t)
            last_spikes[spike_idx] = t
            spike_queue.put(spike_idx)

        arriving_spike = spike_queue.get()
        if arriving_spike:
            V[:, t_idx + 1] += Mo[:, arriving_spike]
            r[arriving_spike, t_idx + 1] += lam_d

            # stdp
            if t > 0:
                delta_ts = t - last_spikes + t_delay
                dW = W(delta_ts)
                Mr[:, arriving_spike:arriving_spike+1] += dW
                Mo[:, arriving_spike:arriving_spike+1] += dW
        Mrs[:, t_idx] = Mr.flatten()

        dx = A @ x_true[:, t_idx] #+ [1,0]
        x_true[:, t_idx + 1] = x_true[:, t_idx] + dt * dx
xhat = D @ r / lam_d


plt.figure('readout')
plt.plot(ts,xhat[0,:], c = 'r',  label='Estimate Dimension 1')
plt.plot(ts, xhat[1,:],c = 'g',  label='Estimate Dimension 2')
plt.plot(ts, x_true[0,:], c ='k', label='True State Dimension 1')
plt.plot(ts, x_true[1,:], c = 'k', label='True State Dimension 2')
plt.legend()
plt.xlabel('Time (Dimensionless)')
plt.ylabel('State')
plt.title('Network Readout vs True State: No Plasticity')


plt.figure('voltages')

plt.plot(ts, V[0,:])
plt.axhline(y=v_th[0])
#plt.imshow(V, aspect='auto')



plt.figure()
for i in range(N):
    plt.plot(Mrs[i,:])
plt.show()
# for each time step,
    # compute & update state
    # if voltage hits threshold, spike
        # update v
        # update r
        # record spike


# get phase trajectories and order parameter from spike times

# question:
    # order parameter trajectory
        # without stdp, report rmse
        # with stdp, rmse


# implement cr & rr stims

    # how does order parameter relate to rmse of error representation for oscillatory dynamics
        # how to vary order parameter?  Use CR/RR
        # average synchrony over trial vs rmse
        # vary each with CR/RR stimulation

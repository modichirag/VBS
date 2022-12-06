import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad
import jax.numpy as jnp
import time

from pmwd.configuration import Configuration
from pmwd.cosmology import SimpleLCDM
from pmwd.lpt import lpt
from pmwd.modes import white_noise, linear_modes
from pmwd.nbody import nbody
from pmwd.scatter import scatter
from pmwd.gravity import rfftnfreq
from pmwd.boltzmann import boltzmann, linear_power
from pmwd.pk import power_spectrum, cross_power

import optax
import sys
sys.path.append('../../hmc/src/')
import algorithms as alg

#####
seed = 0
nc = 64
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1/2. #args.ngrowth
dnoise = 0.1


growth_array = jnp.linspace(0., 1., 32) 
growth_anum = 128

conf0 = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='adaptive')
conf1 = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='mlp')
conf2 = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='rk4')

cosmod = SimpleLCDM(conf0)
cosmod = boltzmann(cosmod,  conf0)

@jit
def simulate_nbody(modes, cosmo):
    '''Run LPT simulation without evaluating growth & tranfer function                                        
    '''
    print("simulate")
    conf = cosmo.conf
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    mesh = scatter(ptcl, conf)
    return mesh


@jit
def log_prob(p0, modes, data, conf):
    print("log prob")
    omegam, As = p0
    cosmo = SimpleLCDM(conf, Omega_m=omegam, A_s_1e9=As)
    cosmo = boltzmann(cosmo, conf)
    lin_modes_c = linear_modes(modes, cosmo, conf)
    mesh = simulate_nbody(lin_modes_c, cosmo)
    chisq = jnp.mean(((data-mesh)/dnoise)**2)
    log_prob = - chisq 
    return log_prob

@jit
def obj(p0, modes, data, conf):
    print("obj")
    return -1. * log_prob(p0, modes, data, conf)


obj_grad = jit(jax.grad(obj, argnums=(0)))
grad_log_prob = jit(jax.grad(log_prob, argnums=(0)))


def gendata(conf, seed):
    modes = white_noise(seed, conf, real=True)
    lin_modes_c = linear_modes(modes, cosmod, conf)
    lin_modes = jnp.fft.irfftn(lin_modes_c)
    dens = simulate_nbody(lin_modes_c, cosmod)
    return modes, lin_modes_c, lin_modes, dens


# A simple update loop.
@jit
def recon_step(p0, modes, opt_state, dens, conf):
    grads = obj_grad(p0, modes, dens, conf)
    updates, opt_state = optimizer.update(grads, opt_state)
    p0 = optax.apply_updates(p0, updates)
    return p0, opt_state


modes, linc, lin, dens = gendata(conf0, seed=10)
data = dens + np.random.normal(0, dnoise, dens.size).reshape(dens.shape).astype(np.float32)
print("Data generated")
print(modes.shape, linc.shape, lin.shape)

omegam, As = 0.15, 1. 
p0 = jnp.array([omegam, As])
print(cosmod.Omega_m, cosmod.A_s_1e9)
print(p0)

for ic, cc in enumerate([conf1, conf2]):
    p0 = jnp.array([omegam, As])
    var = white_noise(1, cc, real=True)*0.1
    g = obj_grad(p0, var, dens, cc);
    print("Compiled gradients")
    traj = []
    traj.append(p0)
    #optimizer
    lr = 0.005
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(p0)
    var, opt_state = recon_step(p0, modes, opt_state, data, cc)

    print("Starting optimization loop")
    # A simple update loop.
    niter  = 1000
    start = time.time()
    for i in range(niter + 1):
        if i%100 == 0: print(i, var)
        var, opt_state = recon_step(var, modes, opt_state, data, cc)
        traj.append(var)
    print("Converged at : ", var)
    end = time.time()
    print("Time taken : ", (end-start)/niter)
    print()
    #np.save('traj%d'%ic, np.array(traj))
    

sys.exit()
#HMC
for i, cc in enumerate([conf1, conf2]):

    omegam, As = 0.15, 1. 
    p0 = jnp.array([omegam, As])
    lp = lambda x: log_prob(x, modes, data, cc)
    lp_g = lambda x: grad_log_prob(x, modes, data, cc)
    print("Call once to compile")
    print(lp(p0))
    print(lp_g(p0))

    def callback(state, testing=False):
        if (state.i % 100 == 0) or testing :
            print(state.i, state.accepts[-1], state.samples[-1])

            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ss = np.array(state.samples)[:]
            ax[0].plot(ss[:, 0])
            ax[1].plot(ss[:, 1])
            ax[0].axhline(cosmod.Omega_m, color='C0')
            ax[1].axhline(cosmod.A_s_1e9, color='C0')
            plt.savefig('samples%d.png'%i)
            plt.close()


            fig, ax = plt.subplots(1, 2, figsize=(8, 3))
            ax[0].hist(ss[:, 0], density=True, bins='auto', alpha=0.8)
            ax[1].hist(ss[:, 1], density=True, bins='auto', alpha=0.8)
            ax[0].axvline(cosmod.Omega_m, color='C0')
            ax[1].axvline(cosmod.A_s_1e9, color='C0')
            ax[0].set_xlim(0, 0.5)
            plt.savefig('hist%d.png'%i)
            plt.close()


    kernel = alg.HMC(log_prob=lp, grad_log_prob=lp_g)
    print('start sampling')
    start = time.time()
    state = kernel.sample(p0, nsamples=5000, burnin=100, step_size=0.01, nleap=20, callback=callback)
    print("Time taken : ", time.time() - start)
    print(state.samples.mean(axis=0))
    print(state.samples.std(axis=0))
    
    if i == 0 : np.save("samples_mlp", state.samples)
    elif i == 1 : np.save("samples_rk4", state.samples)
    print()

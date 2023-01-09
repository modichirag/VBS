import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad, vmap
import jax.numpy as jnp
import optax
import distrax

from pmwd_imports import *

import time, argparse
#$#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--debug', type=int, default=0, help="debug run")
parser.add_argument('--dnoise', type=float, default=1., help='noise level, 1 is shot noise')
parser.add_argument('--reconiter', type=int, default=100, help="number of iterations for reconstruction")
parser.add_argument('--burnin', type=int, default=100, help="number of iterations for burnin")
parser.add_argument('--tadapt', type=int, default=100, help="number of iterations for eps adaptation")
parser.add_argument('--ntrain', type=int, default=10, help="number of training iterations")
parser.add_argument('--thinning', type=int, default=20, help="thinning")
parser.add_argument('--lpsteps1', type=int, default=20, help="min leapfrog steps")
parser.add_argument('--lpsteps2', type=int, default=30, help="max leapfrog steps")
parser.add_argument('--mcmciter', type=int, default=200, help="number of only mcmc iterations")
parser.add_argument('--nplot', type=int, default=20, help="callback after these iterations")
parser.add_argument('--epsadapt', type=int, default=0, help="adapt step size")
parser.add_argument('--order', type=int, default=1, help="ZA or LPT")
parser.add_argument('--nchains', type=int, default=1, help="Number of chains")
args = parser.parse_args()


sys.path.append('../../hmc/src/')
import algorithms as alg
sys.path.append('../src/')
import vbs_tools as tools
from jaxhmc import JaxHMC_vmap, JaxHMC_vmap_args
from vbs_tools import power as power_spectrum
from vbs_utils import gendata, simulate_nbody
from callbacks import callback_hmc_zstep_chains, callback_hmc_qstep_chains

savepath = '//mnt/ceph/users/cmodi/pmwdruns/'
savepath = savepath + 'testvmaps8/'
os.makedirs(savepath, exist_ok=True)
os.makedirs(savepath + '/figs/', exist_ok=True)

#####
seed = 0
nc = 16
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1.0  #0.5 #args.ngrowth
if a_nbody_maxstep > 1 - a_start:
    a_start = 1.
dnoise = 1.0

if args.debug == 1:
    args.nchains = 2
    args.reconiter = 10
    args.burnin = 10
    args.espadapt = 10
    args.mcmciter = 10
    args.tadapt = 10
    args.thinning = 2 
    args.lpsteps1, args.lpsteps2 = 2, 3
    args.nplot = 2

growth_anum = 128
conf = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='mlp', amp_mode='sigma_8')
confdata = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='rk4', amp_mode='sigma_8')
cosmodata = SimpleLCDM_s8(confdata)
cosmodata = boltzmann(cosmodata,  confdata)
print(conf.a_nbody)

@jit
def evolve(p0, modes, conf):
    omegam, sigma_8 = p0
    cosmo = SimpleLCDM_s8(conf, Omega_m=omegam, sigma_8=sigma_8)
    cosmo = boltzmann(cosmo, conf)
    lin_modes_c = linear_modes(modes, cosmo, conf)
    mesh = simulate_nbody(lin_modes_c, cosmo, conf)
    return mesh

@jit
def log_prob_q(p0, modes, data, conf):
    print("log prob q")
    mesh = evolve(p0, modes, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)
    u0, u1 = distrax.Uniform(0.1, 0.5), distrax.Uniform(0.5, 1.)
    log_prior = u0.log_prob(p0[0]) + u1.log_prob(p0[1])
    log_prob = log_lik + log_prior
    return log_prob


@jit
def log_prob_z(modes, p0, data, conf):
    print("log prob z")
    mesh = evolve(p0, modes, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
    log_prior = -0.5 * jnp.sum(modes**2)
    log_prob = log_lik + log_prior
    return log_prob


grad_log_prob_z = jit(jax.grad(log_prob_z, argnums=(0)))
grad_log_prob_q = jit(jax.grad(log_prob_q, argnums=(0)))


def run():

    data_modes, linc, lin, dens, data = gendata(confdata, seed=0, cosmo=cosmodata, dnoise=dnoise,
                                           savepath=savepath)
    true_cosmo = np.array([cosmodata.Omega_m, cosmodata.sigma_8])
    print("Modes mean and std : ", data_modes.mean(), data_modes.std())
    
    print("Data generated")

    omegam, sigma_8 = 0.2, 0.75
    p0 = jnp.array([omegam, sigma_8])
    var = white_noise(99, conf, real=True)
    cosmo = SimpleLCDM_s8(conf, Omega_m=omegam, sigma_8=sigma_8)
    cosmo = boltzmann(cosmo, conf)

    #
    print("###Call once to compile###")
    print()
    print('###Compile z###')
    log_prob_z(var, p0, data, conf)
    grad_log_prob_z(var, p0, data, conf)
    print()
    print("###Compile q###")
    log_prob_q(p0, var, data, conf)
    grad_log_prob_q(p0, var, data, conf)
    print()
    print("###Compiled###")
    
    #Callback
    callback_zstep = lambda state: callback_hmc_zstep_chains(state, parse_args=args,
                                                      conf=conf,
                                                      truth=data_modes,
                                                      savepath=savepath)
    callback_qstep = lambda state: callback_hmc_qstep_chains(state, parse_args=args,
                                                      conf=conf,
                                                      truth=true_cosmo,
                                                      savepath=savepath)

    #Sample        
    print()
    print('###start sampling###')
    zstate = alg.Sampler()
    qstate = alg.Sampler()
    zstep_size = 0.005
    qstep_size = 0.001
    nleap = 30
    thin = 10 
    def qiteration(iq, iz):
        iz = jnp.array(iz, dtype=jnp.float32)
        iq = jnp.array(iq, dtype=jnp.float32)
        lpq = lambda q, z: log_prob_q(jnp.array(q), jnp.array(z), data, conf)
        lpq_g = lambda q, z: grad_log_prob_q(jnp.array(q), jnp.array(z), data, conf)
        kernel_q = JaxHMC_vmap_args(log_prob=lpq, grad_log_prob=lpq_g)
        q, p, acc, Hs, count = kernel_q.step(iq, nleap, qstep_size, iz)
        #print("q acc : ", acc)
        qstate.appends(q, acc, Hs, count)

    def ziteration(iq, iz):
        iq = jnp.array(iq, dtype=jnp.float32)
        iz = jnp.array(iz, dtype=jnp.float32)
        lpz = lambda z, q: log_prob_z(jnp.array(z), jnp.array(q), data, conf)
        lpz_g = lambda z, q: grad_log_prob_z(jnp.array(z), jnp.array(q), data, conf)
        kernel_z = JaxHMC_vmap_args(log_prob=lpz, grad_log_prob=lpz_g)
        z, p, acc, Hs, count = kernel_z.step(iz, nleap, zstep_size, iq)
        #print("z acc : ", acc)
        zstate.appends(np.array(z), acc, Hs, count)


    print("Initialize")
    #p0 = jnp.array([[np.random.normal(0.3, 0.05), np.random.normal(2.0, 0.5)] for _ in range(args.nchains)])    
    p0 = jnp.array([[np.random.uniform(0.15, 0.45), np.random.uniform(0.5, 1.)] for _ in range(args.nchains)])    
    var = jnp.stack([white_noise(i*123, conf, real=True)*0.1 for i in range(args.nchains)])
    print(p0.shape, var.shape)
    print(p0)
    
    print()
    iz, iq = var.copy(), p0.copy()
    print("###Check z iteration###")
    ziteration(iq, iz)
    print()
    print("###Check q iteration###")
    qiteration(iq, iz)
    print()
    zstate.i = 0
    callback_zstep(zstate)
    qstate.i = 0
    callback_qstep(qstate)
    #
    
    iz = zstate.samples[-1]
    iq = qstate.samples[-1]
    start = time.time()
    for i in range(11):
        ziteration(iq, iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)
    print("Time taken for warmup of z : ", time.time() - start)
    
    start = time.time()
    for i in range(101):
        qiteration(iq, iz)
        iq = qstate.samples[-1]
        callback_qstep(qstate)
    print("Time taken for warmup of q : ", time.time() - start)

    for i in range(101):
        ziteration(iq, iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)
        qiteration(iq, iz)
        iq = qstate.samples[-1]
        callback_qstep(qstate)
    print("Time taken for combined warmup : ", time.time() - start)

    #
    print()
    print("### Sampling ###")
    zstate = alg.Sampler()
    qstate = alg.Sampler()

    print()
    start = time.time()
    for i in range(1001):

        ziteration(iq, iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)

        qiteration(iq, iz)
        iq = qstate.samples[-1]
        callback_qstep(qstate)

        if i%100 == 0:
            np.save(savepath + 'qsamples', np.array(qstate.samples))
    print("Time taken : ", time.time() - start)
    

if __name__=="__main__":

    run()

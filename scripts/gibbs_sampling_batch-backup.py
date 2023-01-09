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
args = parser.parse_args()


sys.path.append('../../hmc/src/')
import algorithms as alg
sys.path.append('../src/')
import vbs_tools as tools
from jaxhmc import JaxHMC_vmap
from vbs_tools import power as power_spectrum
from vbs_utils import gendata, simulate_nbody
from callbacks import callback_hmc_zstep, callback_hmc_qstep

savepath = '//mnt/ceph/users/cmodi/pmwdruns/'
savepath = savepath + 'testvmap/'
os.makedirs(savepath, exist_ok=True)
os.makedirs(savepath + '/figs/', exist_ok=True)

#####
seed = 0
nc = 4
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1. #args.ngrowth
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


growth_anum = 32
conf = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='rk4')
confdata = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='rk4')
cosmodata = SimpleLCDM(confdata)
cosmodata = boltzmann(cosmodata,  confdata)

@jit
def evolve(p0, modes, conf):
    omegam, As = p0
    cosmo = SimpleLCDM(conf, Omega_m=omegam, A_s_1e9=As)
    cosmo = boltzmann(cosmo, conf)
    lin_modes_c = linear_modes(modes, cosmo, conf)
    mesh = simulate_nbody(lin_modes_c, cosmo, conf)
    return mesh

@jit
def log_prob_q(p0, modes, data, conf):
    print("log prob q")
    mesh = evolve(p0, modes, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
    log_prob = log_lik
    return log_prob


@jit
def log_prob_z(modes, p0, data, conf):
    print("log prob z")
    mesh = evolve(p0, modes, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
    log_prior = -0.5 * jnp.sum(modes**2)
    log_prob = log_lik + log_prior
    return log_prob

# @jit
# def log_prob_z2(modes, data, cosmo, conf):
#     print("log prob z2")
#     lin_modes_c = linear_modes(modes, cosmo, conf)
#     mesh = simulate_nbody(lin_modes_c, cosmo, conf)
#     log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
#     log_prior = -0.5 * jnp.sum(modes**2)
#     log_prob = log_lik + log_prior
#     return log_prob


#grad_log_prob_z2 = jit(jax.grad(log_prob_z2, argnums=(0)))
grad_log_prob_z = jit(jax.grad(log_prob_z, argnums=(0)))
grad_log_prob_q = jit(jax.grad(log_prob_q, argnums=(0)))


def run():

    modes, linc, lin, dens, data = gendata(confdata, seed=0, cosmo=cosmodata, dnoise=dnoise,
                                           savepath=savepath)
    true_cosmo = np.array([cosmodata.Omega_m, cosmodata.A_s_1e9])

    print("Data generated")
    print(modes.mean(), modes.std())
    print(lin.mean(), lin.std())
    
    omegam, As = 0.2, 1.5 
    #omegam, As = 0.3, 2. 
    p0 = jnp.array([omegam, As])
    var = white_noise(99, conf, real=True)*0.1
    cosmo = SimpleLCDM(conf, Omega_m=omegam, A_s_1e9=As)
    cosmo = boltzmann(cosmo, conf)

    # iz, iq = jnp.stack([var, var*1.1]), jnp.stack([p0, p0])
    # print(iz.shape, iq.shape)
    # f = vmap(lambda a, b: evolve(a, b , conf))
    # toret = f(iq, iz)
    # print(toret.shape)
    
    #
    print("Call once to compile")
    print('compile z')
    log_prob_z(var, p0, data, conf)
    grad_log_prob_z(var, p0, data, conf)
    print()
    print("compile q")
    #log_prob_q(p0, var, data, conf)
    #grad_log_prob_q(p0, var, data, conf)
    print("compiled")
    #jit_cosmo(p0)
    
    #Callback
    callback_zstep = lambda state: callback_hmc_zstep(state, parse_args=args,
                                                      conf=conf,
                                                      truth=modes,
                                                      savepath=savepath)
    callback_qstep = lambda state: callback_hmc_qstep(state, parse_args=args,
                                                      conf=conf,
                                                      truth=true_cosmo,
                                                      savepath=savepath)

    #Sample        
    print()
    print('start sampling')
    zstate = alg.Sampler()
    qstate = alg.Sampler()
    step_size = 0.001
    nleap = 20
    
    def qiteration(iq, iz):
        print("input shape  : ", iq.shape, iz.shape)
        iz = jnp.array(iz, dtype=jnp.float32)
        lpq = lambda x: np.array(log_prob_q(jnp.array(x,dtype=jnp.float32), iz, data, conf))
        lpq_g = lambda x: np.array(grad_log_prob_q(jnp.array(x,dtype=jnp.float32), iz, data, conf))
        kernel_q = alg.HMC(log_prob=lpq, grad_log_prob=lpq_g)
        q, p, acc, Hs, count = kernel_q.step(iq, nleap, step_size)
        print(q.shape)
        #qstate.appends(q, acc, Hs, count)

    # def ziteration(iq, iz):
    #     print("input shape  : ", iq.shape, iz.shape)
    #     iq = jnp.array(iq, dtype=jnp.float32)
    #     lpz = lambda x: np.array(log_prob_z(jnp.array(x, dtype=jnp.float32), iq, data, conf))
    #     lpz_g = lambda x: np.array(grad_log_prob_z(jnp.array(x,dtype=jnp.float32), iq, data, conf))
    #     #kernel_z = alg.HMC(log_prob=lpz, grad_log_prob=lpz_g)
    #     kernel_z = JaxHMC_vmap(log_prob=lpz, grad_log_prob=lpz_g)
    #     q, p, acc, Hs, count = kernel_z.step(iz, nleap, step_size)
    #     zstate.appends(q, acc, Hs, count)

    def ziteration(iq, iz):
        print("input shape  : ", iq.shape, iz.shape)
        iq = jnp.array(iq, dtype=jnp.float32)
        lpz = lambda z, q: log_prob_z(jnp.array(z), jnp.array(q), data, conf)
        lpz_g = lambda z, q: grad_log_prob_z(jnp.array(z), jnp.array(q), data, conf)
        #kernel_z = alg.HMC(log_prob=lpz, grad_log_prob=lpz_g)
        kernel_z = JaxHMC_vmap(log_prob=lpz, grad_log_prob=lpz_g)
        q, p, acc, Hs, count = kernel_z.step(iz, nleap, step_size, iq)
        zstate.appends(q, acc, Hs, count)

        
    print()
    print("test vmap")
    iz, iq = jnp.stack([var, var*1.1]), jnp.stack([p0, p0])
    print(iz.shape, iq.shape)
    ziteration(iq, iz)
    sys.exit()
    #
    iz, iq = var, p0
    ziteration(iq, iz)
    qiteration(iq, iz)
    iz = zstate.samples[-1]
    iq = qstate.samples[-1]

    start = time.time()
    for i in range(100):
        ziteration(iq, iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)
    print("Time taken for warmup of z : ", time.time() - start)
    print()
    start = time.time()
    for i in range(100):
        qiteration(iq, iz)
        iq = qstate.samples[-1]
        callback_qstep(qstate)
    print("Time taken for warmup of q : ", time.time() - start)

    #
    zstate = alg.Sampler()
    qstate = alg.Sampler()
    
    start = time.time()
    for i in range(5000):

        ziteration(iq, iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)

        qiteration(iq, iz)
        iq = qstate.samples[-1]
        callback_qstep(qstate)

    print("Time taken : ", time.time() - start)



if __name__=="__main__":

    run()
    

    
    # def ziteration(iq, iz):

    #     def fun(iq, iz):
    #         print("input shape  : ", iq.shape, iz.shape)
    #         iq = jnp.array(iq, dtype=jnp.float32)
    #         lpz = lambda x: log_prob_z(iq, jnp.array(x, dtype=jnp.float32), data, conf)
    #         lpz_g = lambda x: grad_log_prob_z(iq, jnp.array(x,dtype=jnp.float32), data, conf)
    #         #kernel_z = alg.HMC(log_prob=lpz, grad_log_prob=lpz_g)
    #         kernel_z = JaxHMC_vmap(log_prob=lpz, grad_log_prob=lpz_g)
    #         q = kernel_z.step(iz, nleap, step_size)
    #         #q, p, acc, Hs, count = kernel_z.step(iz, nleap, step_size)
    #         #q = lpz_g(iz)
    #         print("ran, q shape : ", q.shape)
    #         return jnp.array(q)
    #     f = vmap(fun, (0, 0))
    #     toret = f(iq, iz)
    #     print(toret.shape)
    #     #zstate.appends(q, acc, Hs, count)


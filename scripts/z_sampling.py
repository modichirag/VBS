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
parser.add_argument('--nc', type=int, default=32, help="Nmesh")
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
from jaxhmc import JaxHMC_vmap
from vbs_tools import power as power_spectrum
from vbs_utils import gendata, simulate_nbody
from callbacks import callback_hmc_zstep_chains, callback_hmc_qstep_chains

savepath = '//mnt/ceph/users/cmodi/pmwdruns/'
savepath = savepath + 'zsampling_N%d/'%args.nc
os.makedirs(savepath, exist_ok=True)
os.makedirs(savepath + '/figs/', exist_ok=True)

#####
seed = 0
nc = args.nc
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1.0 #0.5 #args.ngrowth
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
cosmo = cosmodata
cosmo = boltzmann(cosmo, conf)

print(conf.a_nbody)

@jit
def evolve(modes, conf):
    lin_modes_c = linear_modes(modes, cosmo, conf)
    mesh = simulate_nbody(lin_modes_c, cosmo, conf)
    return mesh

@jit
def log_prob_z(modes, data, conf):
    print("log prob z")
    mesh = evolve(modes, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
    log_prior = -0.5 * jnp.sum(modes**2)
    log_prob = log_lik + log_prior
    return log_prob


grad_log_prob_z = jit(jax.grad(log_prob_z, argnums=(0)))


def run():

    data_modes, linc, lin, dens, data = gendata(confdata, seed=0, cosmo=cosmodata, dnoise=dnoise,
                                           savepath=savepath)
    print("Modes mean and std : ", data_modes.mean(), data_modes.std())
    
    print("Data generated")

    var = white_noise(99, conf, real=True)

    #
    print("###Call once to compile###")
    print()
    print('###Compile z###')
    log_prob_z(var, data, conf)
    grad_log_prob_z(var, data, conf)
    print()
    print("###Compiled###")
    
    #Callback
    callback_zstep = lambda state: callback_hmc_zstep_chains(state, parse_args=args,
                                                      conf=conf,
                                                      truth=data_modes,
                                                      savepath=savepath)

    #Sample        
    print()
    print('###start sampling###')
    zstate = alg.Sampler()
    zstep_size = np.array([0.005]*args.nchains)
    zepsadapt = alg.DualAveragingStepSize(zstep_size, nadapt=args.epsadapt)
    nleap = 30
    thin = 10 

    
    def ziteration(iz, step_size):
        iz = jnp.array(iz, dtype=jnp.float32)
        lpz = lambda z: log_prob_z(jnp.array(z), data, conf)
        lpz_g = lambda z: grad_log_prob_z(jnp.array(z), data, conf)
        kernel_z = JaxHMC_vmap(log_prob=lpz, grad_log_prob=lpz_g)
        z, p, acc, Hs, count = kernel_z.step(iz, nleap, step_size)
        #print("z acc : ", acc)
        zstate.appends(np.array(z), acc, Hs, count)
        step_size = zepsadapt(zstate.i, np.exp(Hs[0]-Hs[1]))
        return step_size

    print("Initialize")
    var = jnp.stack([white_noise(i*123, conf, real=True)*0.1 for i in range(args.nchains)])
    print(var.shape)
    
    print()
    iz = var.copy()
    print("###Check z iteration###")
    ziteration(iz)
    print()
    zstate.i = 0
    callback_zstep(zstate)
    
    iz = zstate.samples[-1]
    start = time.time()
    for i in range(11):
        ziteration( iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)
    print("Time taken for warmup of z : ", time.time() - start)
    

    for i in range(101):
        ziteration(iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)
    print("Time taken for combined warmup : ", time.time() - start)

    #
    print()
    print("### Sampling ###")
    zstate = alg.Sampler()
    zstate.ps = []
    zstate.px = []
    
    print()
    start = time.time()
    for i in range(10001):

        ziteration(iz)
        iz = zstate.samples[-1]
        callback_zstep(zstate)

        if i%100 == 0:
            np.save(savepath + 'ps', np.array(zstate.ps))
            np.save(savepath + 'px', np.array(zstate.px))
            np.save(savepath + 'z', np.array(zstate.samples)[::thin])
    print("Time taken : ", time.time() - start)
    

if __name__=="__main__":

    run()

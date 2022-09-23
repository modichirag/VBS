import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad
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
from vbs_tools import power as power_spectrum
from vbs_utils import gendata, simulate_nbody
from callbacks import callback_hmc_zstep

savepath = '//mnt/ceph/users/cmodi/pmwdruns/'
savepath = savepath + 'test/'
os.makedirs(savepath, exist_ok=True)
os.makedirs(savepath + '/figs/', exist_ok=True)

#####
seed = 0
nc = 64
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1/2. #args.ngrowth
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
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='mlp')
confdata = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, growth_anum=growth_anum, growth_mode='rk4')
cosmodata = SimpleLCDM(confdata)
cosmodata = boltzmann(cosmodata,  confdata)


@jit
def log_prob_z(modes, data, cosmo, conf):
    print("log prob_z")
    lin_modes_c = linear_modes(modes, cosmo, conf)
    mesh = simulate_nbody(lin_modes_c, cosmo, conf)
    log_lik = -0.5 * jnp.sum(((data-mesh)/dnoise)**2)     
    log_prior = -0.5 * jnp.sum(modes**2)
    log_prob = log_lik + log_prior
    return log_prob

grad_log_prob_z = jit(jax.grad(log_prob_z, argnums=(0)))


#########
def run():

    modes, linc, lin, dens, data = gendata(confdata, seed=0, cosmo=cosmodata, dnoise=dnoise,
                                           savepath=savepath)
    print("Data generated")
    print(modes.mean(), modes.std())
    print(lin.mean(), lin.std())
    
    var = white_noise(99, conf, real=True)*0.1
    #
    lpz = lambda x: log_prob_z(x, data, cosmodata, conf)
    lpz_g = lambda x: grad_log_prob_z(x, data, cosmodata, conf)
    print("Call once to compile")
    print(lpz(var))
    print()
    print(lpz_g(var).shape)
    print()
    
    #Callback
    callback_zstep = lambda state: callback_hmc_zstep(state, parse_args=args,
                                                      conf=conf,
                                                      truth=modes,
                                                      savepath=savepath)
    #Sample        
    kernel = alg.HMC(log_prob=lpz, grad_log_prob=lpz_g)
    print('start sampling')
    start = time.time()
    state = kernel.sample(var, nsamples=args.mcmciter,
                          burnin=args.burnin, step_size=1e-2,
                          nleap= 20, #np.random.randint(args.lpsteps1, args.lpsteps2),
                          callback=callback_zstep, epsadapt=args.epsadapt)
    print("Time taken : ", time.time() - start)

    np.save(savepath + "samples", state.samples)
    


if __name__=="__main__":

    run()
    

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


seed = 0
nc = 128
cell_size = 4.
box_size = np.float(nc*cell_size)
a_start = 0.1
a_nbody_maxstep = 1/5. #args.ngrowth                                                                                                                                                                                                                                                                                     

conf = Configuration(ptcl_spacing=cell_size, ptcl_grid_shape=(nc,)*3, \
                     a_start=a_start, a_nbody_maxstep=a_nbody_maxstep, )
cosmo = SimpleLCDM(conf)
cosmo = boltzmann(cosmo, conf)
kvec = rfftnfreq(conf.ptcl_grid_shape, conf.ptcl_spacing, dtype=conf.float_dtype)
k = jnp.sqrt(sum(k**2 for k in kvec))
plinmesh = linear_power(k, 1.0, cosmo, conf)


@jit
def simulate_nbody(modes, cosmo):
    '''Run LPT simulation without evaluating growth & tranfer function                                                                                                                                                                                                                                                        
    '''
    conf = cosmo.conf
    ptcl, obsvbl = lpt(modes, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    mesh = scatter(ptcl, conf)
    return mesh


@jit
def obj(var, data, cosmo):
#     cosmo = boltzmann(cosmo, conf)
    conf = cosmo.conf
    modes = linear_modes(var, cosmo, conf)
    mesh = simulate_nbody(modes, cosmo)
    chisq = jnp.mean((data-mesh)**2)
    
    prior = jnp.mean(var**2)
    loss = chisq + prior
    return loss

obj_grad = jit(jax.grad(obj, argnums=(0)))


def gendata(seed=0):
    modes = white_noise(seed, conf, real=True)
    lin_modes_c = linear_modes(modes, cosmo, conf)
    lin_modes = jnp.fft.irfftn(lin_modes_c)
    dens = simulate_nbody(lin_modes_c, cosmo)
    return modes, dens


# A simple update loop.
@jit
def recon_step(var, opt_state, dens, cosmo):
    grads = obj_grad(var, dens, cosmo)
    updates, opt_state = optimizer.update(grads, opt_state)
    var = optax.apply_updates(var, updates)
    return var, opt_state



modes, dens = gendata()
data = dens + np.random.normal(0, 1, dens.size).reshape(dens.shape).astype(np.float32)
print("Data generated")

var = white_noise(1, conf, real=True)*0.1
obj_grad(var, dens, cosmo);
print("Compiled gradients")
#optimizer
lr = 0.01
optimizer = optax.adam(lr)
opt_state = optimizer.init(var)
var, opt_state = recon_step(var, opt_state, data, cosmo)

print("Starting optimization loop")
# A simple update loop.
for i in range(501):
    if i%100 == 0: print(i)
    var, opt_state = recon_step(var, opt_state, data, cosmo)


k, pk0 = power_spectrum(var, boxsize=box_size)
k, pk1 = power_spectrum(modes, boxsize=box_size)
k, rc = cross_power(var, modes, boxsize=box_size)

plt.plot(k[1:], rc[1:]/(pk0[1:] * pk1[1:])**0.5)
plt.grid()
plt.savefig('rc.png')

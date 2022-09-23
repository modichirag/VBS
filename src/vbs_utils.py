import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys, os, time

import jax
from jax import jit, custom_vjp, ensure_compile_time_eval, grad
import jax.numpy as jnp

from pmwd_imports import *

##My imports
sys.path.append('../../hmc/src/')
import algorithms as alg

import vbs_tools as tools
from vbs_tools import power as power_spectrum

#####
@jit
def simulate_nbody(lin_modes_c, cosmo, conf):
    '''Run LPT simulation without evaluating growth & tranfer function               
    '''
    print("nbody")
    ptcl, obsvbl = lpt(lin_modes_c, cosmo, conf)
    ptcl, obsvbl = nbody(ptcl, obsvbl, cosmo, conf)
    mesh = scatter(ptcl, conf)
    return mesh


def gendata(conf, seed, cosmo, dnoise=1., savepath=None):
    modes = white_noise(seed, conf, real=True)
    lin_modes_c = linear_modes(modes, cosmo, conf)
    lin_modes = jnp.fft.irfftn(lin_modes_c, s=conf.ptcl_grid_shape, norm='ortho')
    dens = simulate_nbody(lin_modes_c, cosmo, conf)
    noise = np.random.normal(0, dnoise, dens.size).reshape(dens.shape).astype(np.float32)
    data = dens + noise
    if savepath is not None:
        np.save(savepath + "modes", modes)
        np.save(savepath + "linc", lin_modes_c)
        np.save(savepath + "lin", lin_modes)
        np.save(savepath + "dens", dens)
        np.save(savepath + "data", data)

        box_size = conf.box_size[0]
        k, pk = power_spectrum(1+modes, boxsize=box_size)
        plt.plot(k, pk, label='modes')
        k, pk = power_spectrum(1+lin_modes, boxsize=box_size)
        plt.plot(k, pk, label='linear')
        k, pk = power_spectrum(dens, boxsize=box_size)
        plt.plot(k, pk, label='final')
        k, pk = power_spectrum(data, boxsize=box_size)
        plt.plot(k, pk, label='data')
        plt.legend()
        plt.loglog()
        plt.grid(which='both', alpha=0.5)
        plt.savefig(savepath + 'dataps.png')
        plt.close()

        #
        fig, axar = plt.subplots(2, 2, figsize=(8, 8))
        im = axar[0, 0].imshow(modes.sum(axis=0))
        plt.colorbar(im, ax=axar[0, 0])
        axar[0, 0].set_title('Modes')
        im = axar[0, 1].imshow(lin_modes.sum(axis=0))
        plt.colorbar(im, ax=axar[0, 1])
        axar[0, 1].set_title('Linear')
        im = axar[1, 0].imshow(dens.sum(axis=0))
        plt.colorbar(im, ax=axar[1, 0])
        axar[1, 0].set_title('Final')
        im = axar[1, 1].imshow(data.sum(axis=0))
        plt.colorbar(im, ax=axar[1, 1])
        axar[1, 1].set_title('Data')
        plt.savefig(savepath + 'dataim.png')
        plt.close()
        
    return modes, lin_modes_c, lin_modes, dens, data



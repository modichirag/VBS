import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.append('../src/')
import vbs_tools as tools
from vbs_tools import power as power_spectrum
#import diagnostics as dg



def callback_hmc_zstep(state, testing=False, **kwargs):

    args = kwargs['parse_args']
    
    if (state.i % args.nplot == 0) or testing :
        print("z iteration : ", state.i)
        acc = np.array(state.accepts)
        print("accepted fraction of zs : ", (acc == 1).sum()/acc.size)

        #
        conf = kwargs['conf']
        box_size = conf.box_size[0]
        truth = kwargs['truth']
        
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        ss = np.array(state.samples)
        ss = ss[-1]
        k, pk0 = power_spectrum(1+ss, boxsize=box_size)
        k, pk1 = power_spectrum(1+truth, boxsize=box_size)
        _, px = power_spectrum(1+ss, 1+truth, boxsize=box_size)

        ax[0].plot(k[1:], px[1:]/(pk0[1:] * pk1[1:])**0.5)
        ax[1].plot(k[1:], (pk0[1:] / pk1[1:])**0.5)
        ax[2].plot(k[1:], pk0[1:])
        ax[2].plot(k[1:], pk1[1:], '--')
        for axis in ax:
            axis.set_xscale('log')
            axis.grid(which='both', alpha=0.5)
        plt.savefig(kwargs['savepath'] + 'figs/rc-%05d.png'%(state.i))
        plt.close()


def callback_hmc_qstep(state, testing=False, **kwargs):

    args = kwargs['parse_args']
    
    if (state.i % args.nplot == 0) or testing :
        print("qstep iteration : ", state.i)
        acc = np.array(state.accepts)
        print("accepted fraction of qs: ", (acc == 1).sum()/acc.size)

        #
        conf = kwargs['conf']
        box_size = conf.box_size[0]
        truth = kwargs['truth']
        
        ss = np.array(state.samples)
        fig = corner(ss, truth)
        plt.savefig(kwargs['savepath'] + 'figs/qcorner-%05d.png'%(state.i))
        plt.close()

        ndim = ss.shape[-1]
        fig, ax = plt.subplots(1, ndim, figsize=(ndim*3.5, 3))
        for i in range(ndim):
            ax[i].plot(ss[:, i])
            ax[i].axhline(truth[i], color='k', ls='--')
        plt.savefig(kwargs['savepath'] + 'figs/qiter-%05d.png'%(state.i))
        plt.close()

        
def callback_hmc_zstep_chains(state, testing=False, **kwargs):
    #Expected shape of samples (ss) in [nsamples, nchains, nc, nc, nc]

    args = kwargs['parse_args']
    conf = kwargs['conf']
    box_size = conf.box_size[0]
    truth = kwargs['truth']

    #Save power spectrum
    try: state.ps
    except:
        state.ps, state.px = [], []

    ss = state.samples[-1]
    k, pt = power_spectrum(1+truth, boxsize=box_size)
    pk = [power_spectrum(1+ss[j], boxsize=box_size)[1] for j in range(args.nchains)]
    px = [power_spectrum(1+ss[j], 1+truth, boxsize=box_size)[1] for j in range(args.nchains)]
    state.ps.append(pk)
    state.px.append(px)

    if (state.i % args.nplot == 0) or testing :
        print("z iteration : ", state.i)
        acc = np.array(state.accepts)
        print(acc.shape)
        print("accepted fraction of zs :", ((acc == 1).sum(axis=0)/acc.shape[0]))

        #
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))

        for j in range(args.nchains):
            ax[0].plot(k[1:], px[j][1:]/(pk[j][1:] * pt[1:])**0.5)
            ax[1].plot(k[1:], (pk[j][1:] / pt[1:])**0.5)
            ax[2].plot(k[1:], pk[j][1:])
            ax[2].plot(k[1:], pt[1:], 'k--')
            
        ax[0].set_ylabel('$r_c$', fontsize=12)
        ax[1].set_ylabel('$t_f$', fontsize=12)
        ax[2].set_ylabel('$P(k)$', fontsize=12)
        ax[0].set_ylim(0, 1.1)
        ax[1].set_ylim(0, 2)
        ax[0].axhline(1, color='k', lw=0.5)
        ax[1].axhline(1, color='k', lw=0.5)
        for axis in ax:
            axis.set_xscale('log')
            axis.grid(which='both', alpha=0.5)
        plt.savefig(kwargs['savepath'] + 'figs/ziter-%05d.png'%(state.i))
        plt.close()


def callback_hmc_qstep_chains(state, testing=False, **kwargs):
    #Expected shape of samples (ss) in [nsamples, nchains, ndim]

    args = kwargs['parse_args']
    
    if (state.i % args.nplot == 0) or testing :
        print("qstep iteration : ", state.i)
        acc = np.array(state.accepts)
        print("accepted fraction of qs :", ((acc == 1).sum(axis=0)/acc.shape[0]))

        #
        conf = kwargs['conf']
        box_size = conf.box_size[0]
        truth = kwargs['truth']
        
        ss = np.array(state.samples)
        for j in range(args.nchains):
            fig = corner(ss[:, j], truth)
            plt.savefig(kwargs['savepath'] + 'figs/qcorner%d-%05d.png'%(j, state.i))
            plt.close()

        ndim = ss.shape[-1]
        fig, ax = plt.subplots(1, ndim, figsize=(ndim*3.5, 3))
        for j in range(args.nchains):
            for i in range(ndim):
                ax[i].plot(ss[:, j, i])
                ax[i].axhline(truth[i], color='k', ls='--')
        plt.savefig(kwargs['savepath'] + 'figs/qiter-%05d.png'%(state.i))
        plt.close()



def corner(samples, pp):
    ndim = samples.shape[1]
    fig, ax = plt.subplots(ndim, ndim, figsize=(ndim*2, ndim*2), sharex=False, sharey=False)
    for i in range(ndim):
        for j in range(ndim):
            if i == j: 
                ax[i, j].hist(samples[:, i], bins='auto', density=True)
                mu, sig = samples[:, i].mean(), samples[:, i].std()
                ax[i, j].axvline(mu, ls="-", color="r", label="%0.3f"%mu, lw=0.7)
                ax[i, j].axvline(mu + sig, ls=":", color="r",lw=0.7)
                ax[i, j].axvline(mu - sig, ls=":", color="r", lw=0.7)
                ax[i, j].axvline(pp[i], ls="-", color="k", label="%0.3f"%pp[i], lw=0.7)
                ax[i, j].legend()
                #ax[i, j].set_title(names[i])
            elif i>j: 
                ax[i, j].plot(samples[:, i], samples[:, j], '.')
                ax[i, j].axvline(pp[i], ls="--", color="k")
                ax[i, j].axhline(pp[j], ls="--", color="k")
                ax[i, j].axvline(samples[:, i].mean(), ls="-", color="r", lw=0.7)
                ax[i, j].axhline(samples[:, j].mean(), ls="-", color="r", lw=0.7)
            else: ax[i, j].set_visible(False)
    plt.tight_layout()
    return fig



# def callback(model, ic, bs, losses=None):
    
#     fig, ax = plt.subplots(1, 6, figsize=(15, 3))
#     im = ax[0].imshow(ic[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[0])
#     ax[0].set_title('Truth')
#     #
#     sample = model.sample_linear
#     im = ax[1].imshow((sample).numpy()[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[1])
#     ax[1].set_title('Sample')
#     #
#     diff = sample - ic
#     im = ax[2].imshow(diff.numpy()[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[2])
#     ax[2].set_title('Differnce')


#     #2pt functions
#     k, p0 = tools.power(ic[0]+1, boxsize=bs)
#     ps, rc, ratios = [], [], []
#     for i in range(20):
#         sample = model.sample_linear
#         i0 = (sample).numpy()[0]
#         k, p1 = tools.power(i0+1, boxsize=bs)
#         k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
#         ps.append([p1, p1x])
#         rc.append(p1x/(p1*p0)**0.5)
#         ratios.append((p1/p0)**0.5)
#     rc = np.array(rc)
#     ratios = np.array(ratios)
    
#     ax = ax[3:]
#     ax[0].plot(k, rc.T, 'C1', alpha=0.2)
#     ax[0].plot(k, rc.mean(axis=0))
#     ax[0].semilogx()
#     ax[0].set_ylim(0., 1.05)
#     ax[0].set_title('$r_c$', fontsize=12)
    
#     ax[1].plot(k, ratios.T, 'C1', alpha=0.2)
#     ax[1].plot(k, ratios.mean(axis=0))
#     ax[1].semilogx()
#     ax[1].set_ylim(0.8, 1.2)
#     ax[1].set_title('$t_f$', fontsize=12)
    
# #     if losses is not None: ax[2].plot(losses)
#     if losses is not None: 
#         losses = -1. * np.array(losses)
#         ax[2].plot(losses[:, 0], label='-logl')
#         ax[2].plot(losses[:, 1], label='-logp')
#         ax[2].plot(losses[:, 2], label='-logq')
#         ax[2].plot(losses[:, 3], 'k', label='-elbo')
#     ax[2].loglog()
#     ax[2].set_title('-ELBO', fontsize=12)
#     ax[2].legend()
#     for axis in ax: axis.grid(which='both')
    
#     plt.tight_layout()
#     return fig




# def callback_fvi(model, ic, bs, losses=None, zoomin=True, linear=True):
    
#     fig, ax = plt.subplots(1, 6, figsize=(15, 3))
#     im = ax[0].imshow(ic[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[0])
#     ax[0].set_title('Truth')
#     #
#     if linear: 
#         sample = model.sample_linear
#     else :  
#         sample = model.q.sample(1)
#     im = ax[1].imshow((sample).numpy()[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[1])
#     ax[1].set_title('Sample')
#     #
#     diff = sample - ic
#     im = ax[2].imshow(diff.numpy()[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[2])
#     ax[2].set_title('Differnce')


#     #2pt functions
#     k, p0 = tools.power(ic[0]+1, boxsize=bs)
#     ps, rc, ratios = [], [], []
#     for i in range(20):
#         if linear : 
#             sample = model.sample_linear
#         else : 
#             sample = model.q.sample(1)
#         i0 = (sample).numpy()[0]
#         k, p1 = tools.power(i0+1, boxsize=bs)
#         k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
#         ps.append([p1, p1x])
#         rc.append(p1x/(p1*p0)**0.5)
#         ratios.append((p1/p0)**0.5)
#     rc = np.array(rc)
#     ratios = np.array(ratios)
    
#     ax = ax[3:]
#     ax[0].plot(k, rc.T, 'C1', alpha=0.2)
#     ax[0].plot(k, rc.mean(axis=0))
#     ax[0].semilogx()
#     ax[0].set_ylim(0., 1.05)
#     ax[0].set_title('$r_c$', fontsize=12)
    
#     ax[1].plot(k, ratios.T, 'C1', alpha=0.2)
#     ax[1].plot(k, ratios.mean(axis=0))
#     ax[1].semilogx()
#     if zoomin: ax[1].set_ylim(0.8, 1.2)
#     else: ax[1].set_ylim(0.0, 1.5)
#     ax[1].set_title('$t_f$', fontsize=12)
    
#     ax[2].plot(losses)
#     ax[2].loglog()
#     ax[2].set_title('-logq', fontsize=12)
#     ax[2].legend()
#     for axis in ax: axis.grid(which='both')
    
#     plt.tight_layout()
#     return fig




# def callback_sampling(samples, ic, bs):
    
#     fig, axar = plt.subplots(2, 3, figsize=(12, 8))
#     ax = axar[0]
#     im = ax[0].imshow(ic[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[0])
#     ax[0].set_title('Truth')
#     #
#     sample = samples[np.random.randint(len(samples))].numpy()
#     im = ax[1].imshow((sample)[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[1])
#     ax[1].set_title('Sample')
#     #
#     diff = sample - ic
#     im = ax[2].imshow(diff[0].sum(axis=0))
#     plt.colorbar(im, ax=ax[2])
#     ax[2].set_title('Differnce')


#     #2pt functions
#     k, p0 = tools.power(ic[0]+1, boxsize=bs)
#     ps, rc, ratios = [], [], []
#     for i in range(len(samples)):
#         sample = samples[i].numpy()
#         if len(sample.shape) == 4: 
#             for j in range(sample.shape[0]):
#                 i0 = (sample)[j]
#                 k, p1 = tools.power(i0+1, boxsize=bs)
#                 k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
#                 ps.append([p1, p1x])
#                 rc.append(p1x/(p1*p0)**0.5)
#                 ratios.append((p1/p0)**0.5)
#         elif len(sample.shape) == 3:
#             i0 = sample.copy()
#             k, p1 = tools.power(i0+1, boxsize=bs)
#             k, p1x = tools.power(i0+1, ic[0]+1, boxsize=bs)
#             ps.append([p1, p1x])
#             rc.append(p1x/(p1*p0)**0.5)
#             ratios.append((p1/p0)**0.5)
#     rc = np.array(rc)
#     ratios = np.array(ratios)
    
#     ax = axar[1]
#     ax[0].plot(k, rc.T, alpha=0.3)
#     ax[0].plot(k, rc.mean(axis=0))
#     ax[0].semilogx()
#     ax[0].set_ylim(0., 1.05)
#     ax[0].set_title('$r_c$', fontsize=12)
    
#     ax[1].plot(k, ratios.T, alpha=0.3)
#     ax[1].plot(k, ratios.mean(axis=0))
#     ax[1].semilogx()
#     ax[1].set_ylim(0.8, 1.2)
#     ax[1].set_title('$t_f$', fontsize=12)
        
#     ax[2].plot(k, p0, 'k', alpha=0.8)
#     for ip in ps:
#         ax[2].plot(k, ip[0], alpha=0.3)
#     ax[2].loglog()
    
#     for axis in ax: axis.grid(which='both')
#     plt.tight_layout()
#     return fig



# def datafig(ic, fin, data, bs, dnoise, shotnoise=None):
#     nc = ic.shape[-1]
#     k, pic = tools.power(ic[0], boxsize=bs)
#     k, pf = tools.power(fin[0], boxsize=bs)
#     k, pd = tools.power(data[0], boxsize=bs)
#     k, pn = tools.power(1+data[0]-fin[0], boxsize=bs)
#     if dnoise is not None:
#         k, pn2 = tools.power(1+np.random.normal(0, dnoise, nc**3).reshape(fin.shape)[0], boxsize=bs)

#     # plt.plot(k, pd/pf)
#     # plt.semilogx()
#     fig, axar = plt.subplots(2, 2, figsize=(8, 8))

#     im = axar[0, 0].imshow(ic[0].sum(axis=0))
#     plt.colorbar(im, ax=axar[0, 0])
#     axar[0, 0].set_title('IC')
#     im = axar[0, 1].imshow(fin[0].sum(axis=0))
#     plt.colorbar(im, ax=axar[0, 1])
#     axar[0, 1].set_title('Final')
#     im = axar[1, 0].imshow(data[0].sum(axis=0))
#     plt.colorbar(im, ax=axar[1, 0])
#     axar[1, 0].set_title('Data')
#     ax = axar[1]
#     ax[1].plot(k, pic, label='IC')
#     ax[1].plot(k, pf, label='Final')
#     ax[1].plot(k, pd, label='Data')
#     ax[1].plot(k, pn, label='Noise')
#     ax[1].axhline((bs**3/nc**3))
#     if shotnoise is not None: ax[1].axhline(shotnoise, color='k', ls="--")

#     ax[1].loglog()
#     ax[1].grid(which='both')
#     ax[1].legend()
#     ax[1].set_xlabel('k (h/Mpc)')
#     ax[1].set_ylabel('P(k)')
#     plt.suptitle('LPT: Boxsize=%d, Nmesh=%d'%(bs, nc))
#     return fig



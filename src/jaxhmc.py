import numpy as np
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
import jax
key = jax.random.PRNGKey(0)

import sys
sys.path.append('../../hmc/src/')
import algorithms as alg
HMC = alg.HMC



#####
# class JaxHMC_vmap2(HMC):
    
#     def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None):
#         super(JaxHMC_vmap, self).__init__(log_prob, grad_log_prob, log_prob_and_grad, invmetric_diag=invmetric_diag)        
#         self.leapfrog_vmap = vmap(self.leapfrog)
#         self.H_vmap = vmap(self.H)

#     def metropolis(self, qp0, qp1, H0, H1):
#         q0, p0 = qp0
#         q1, p1 = qp1
#         prob = np.exp(H0 - H1)
#         acc  = []
#         q2, p2, V2 = [], [], []
#         for i in range(q0.shape[0]):
#             iprob = prob[i]
#             accept = 1
#             if np.isnan(iprob) or (q0-q1).sum()==0: 
#                 accept = 2
#             elif np.random.uniform(0., 1., size=1) > min(1., iprob):
#                 accept = 0
#             acc.append(accept)
#             if accept == 1:
#                 q0[i] = q1[i].copy()
#                 p0[i] = p1[i].copy()
#         return q0, p0, np.array(acc), np.array([H0, H1], dtype=np.float32)

    
#     def step(self, q, nleap, step_size):

#         self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
#         batch = q.shape[0]
#         p = jnp.array(np.random.normal(size=q.size).reshape(q.shape).astype(q.dtype) * self.metricstd)
#         nleap, step_size = jnp.array([nleap]*batch), jnp.array([step_size]*batch)

#         print('step qp : ', q.shape, p.shape)
#         q1, p1 = self.leapfrog_vmap(q, p, nleap, step_size)
#         print('step qp1 : ', q1.shape, p1.shape)
#         H0 = self.H_vmap(q, p)
#         H1 = self.H_vmap(q1, p1)
#         q, p, accepted, Hs = self.metropolis([q, p], [q1, p1], H0, H1)
#         return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]

    

class JaxHMC_vmap():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
        self.returnV = returnV
        self.leapfrog_vmap = vmap(self.leapfrog, in_axes=(0, 0, None, None))
        self.H_vmap = vmap(self.H, in_axes=(0, 0))
    
    def V_g(self, x):
        self.Vgcount += 1
        return self.grad_log_prob(x)*-1.
        
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q)
            return V, V + self.KE(p)
        else:
            return self.V(q) + self.KE(p)

    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p
        except Exception as e:
            print("Exception occured in leapfrog : \n", e)
            return q0, p0


    def metropolis(self, qp0, qp1, H0, H1):
        q0, p0 = qp0
        q1, p1 = qp1

        prob = np.exp(H0 - H1)
        acc  = []
        for i in range(q0.shape[0]):
            iprob = prob[i]
            accept = 1
            if np.isnan(iprob) or (q0[i]-q1[i]).sum()==0: 
                accept = 2
            elif np.random.uniform(0., 1., size=1) > min(1., iprob):
                accept = 0
            acc.append(accept)
            if accept == 1:
                q0 = q0.at[i].set(q1[i])
                p0 = p0.at[i].set(p1[i])
        return q0, p0, np.array(acc), np.array([H0, H1], dtype=np.float32)



    def step(self, q, nleap, step_size):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = jnp.array(np.random.normal(size=q.size).reshape(q.shape).astype(q.dtype) * self.metricstd)
        q1, p1 = self.leapfrog_vmap(q, p, nleap, step_size)
        H0 = self.H_vmap(q, p)
        H1 = self.H_vmap(q1, p1)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1], H0, H1)
        return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]



class JaxHMC_vmap_args():
    
    def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x, args : self.log_prob(x, args)*-1.
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        self.metricstd = self.invmetric_diag**-0.5

        self.KE = lambda p: 0.5*(p**2 * self.invmetric_diag).sum()
        self.KE_g = lambda p: p * self.invmetric_diag
        self.returnV = returnV
        self.leapfrog_vmap = vmap(self.leapfrog, in_axes=(0, 0, None, None, 0))
        self.H_vmap = vmap(self.H, in_axes=(0, 0, 0))
    
    def V_g(self, q, args):
        self.Vgcount += 1
        return self.grad_log_prob(q, args)*-1.
        
    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()

    def unit_norm_KE_g(self, p):
        return p

    def H(self, q, p, args):
        self.Hcount += 1
        if self.returnV:
            V = self.V(q, args)
            return V, V + self.KE(p)
        else: return self.V(q, args) + self.KE(p)

    def leapfrog(self, q, p, N, step_size, args):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q, args) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q, args) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q, args) 
            return q, p
        except Exception as e:
            print("Exception occured in leapfrog : \n", e)
            return q0, p0


    def metropolis(self, qp0, qp1, H0, H1):
        q0, p0 = qp0
        q1, p1 = qp1

        prob = np.exp(H0 - H1)
        acc  = []
        for i in range(q0.shape[0]):
            iprob = prob[i]
            accept = 1
            if np.isnan(iprob) or (q0[i]-q1[i]).sum()==0: 
                accept = 2
            elif np.random.uniform(0., 1., size=1) > min(1., iprob):
                accept = 0
            acc.append(accept)
            if accept == 1:
                q0 = q0.at[i].set(q1[i])
                p0 = p0.at[i].set(p1[i])
        return q0, p0, np.array(acc), np.array([H0, H1], dtype=np.float32)



    def step(self, q, nleap, step_size, args):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = jnp.array(np.random.normal(size=q.size).reshape(q.shape).astype(q.dtype) * self.metricstd)
        q1, p1 = self.leapfrog_vmap(q, p, nleap, step_size, args)
        H0 = self.H_vmap(q, p, args)
        H1 = self.H_vmap(q1, p1, args)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1], H0, H1)
        return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]
    
    
    
# class PyHMC_batch():
    
#     def __init__(self, log_prob, grad_log_prob, returnV=False, invmetric_diag=None):

#         self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
#         self.V = lambda x : self.log_prob(x)*-1.
#         #self.V_g = lambda x : self.grad_log_prob(x)*-1.
#         self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0

#         if invmetric_diag is None: self.invmetric_diag = 1.
#         else: self.invmetric_diag = invmetric_diag
#         self.metricstd = self.invmetric_diag**-0.5

#         self.returnV = returnV
    
#     def KE(self, p):
#         #ke = np.sum(0.5*(p**2), axis=(-3, -2, -1))
#         ke = np.sum(0.5*(p**2), axis=tuple(range(1, len(p.shape))))
#         #ke = np.sum(0.5*(p**2 * self.invmetric_diag), axis=(-3, -2, -1))
#         return ke

#     def KE_g(self, p):
#         return p 

#     def V_g(self, x):
#         self.Vgcount += 1
#         return self.grad_log_prob(x)*-1.
       

#     def H(self, q, p):
#         self.Hcount += 1
#         if self.returnV:
#             V = self.V(q)
#             return V, V + self.KE(p)
#         else: return self.V(q) + self.KE(p)

#     def leapfrog(self, q, p, N, step_size):
#         self.leapcount += 1 
#         q0, p0 = q, p
#         p = p - 0.5*step_size * self.V_g(q) 
#         for i in range(N-1):
#             q = q + step_size * self.KE_g(p)
#             p = p - step_size * self.V_g(q) 
#         q = q + step_size * self.KE_g(p)
#         p = p - 0.5*step_size * self.V_g(q) 
#         return q, p


#     def metropolis(self, qp0, qp1):
#         q0, p0 = qp0
#         q1, p1 = qp1
#         if self.returnV:
#             V0, H0 = self.H(q0, p0)
#             V1, H1 = self.H(q1, p1)
#         else:
#             H0 = self.H(q0, p0)
#             H1 = self.H(q1, p1)
#             V0, V1 = [None]*q0.shape[0], [None]*q0.shape[0]

#         prob = np.exp(H0 - H1)
#         acc  = []
#         q2, p2, V2 = [], [], []
#         for i in range(q0.shape[0]):
#             iprob = prob[i]
#             accept = 1
#             if np.isnan(iprob) or (q0-q1).sum()==0: 
#                 accept = 2
#             elif np.random.uniform(0., 1., size=1) > min(1., iprob):
#                 accept = 0
#             acc.append(accept)
#             if accept == 1:
#                 q0[i] = q1[i].copy()
#                 p0[i] = p1[i].copy()
#                 V0[i] = V1[i]                
#         return q0, p0, np.array(acc), np.array([H0, H1, V0], dtype=np.float32)

#     def hmc_step(self, q, N, step_size):

#         step_sizeb = step_size.copy()
#         for idim in range(1, len(q.shape)): step_sizeb = np.expand_dims(step_sizeb, -1)
#         p = np.random.normal(size=q.size).reshape(q.shape) #* self.metricstd
#         q1, p1 = self.leapfrog(q, p, N, step_sizeb)
#         q, p, accepted, energy = self.metropolis([q, p], [q1, p1])    
#         return q, p, accepted, energy, [self.Hcount, self.Vgcount, self.leapcount]



#     # def metropolis(self, qp0, qp1, args):
#     #     q0, p0 = qp0
#     #     q1, p1 = qp1
#     #     if self.returnV:
#     #         V0, H0 = self.H(q0, p0, args)
#     #         V1, H1 = self.H(q1, p1, args)
#     #     else:
#     #         H0 = self.H(q0, p0, args)
#     #         H1 = self.H(q1, p1, args)
#     #         V0, V1 = None, None
#     #     prob = jnp.exp(H0 - H1)
#     #     #if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
#     #     def _error(prob, q0, p0, q1, p1, H0, H1, V0, V1):
#     #         return q0, p0, 2., [H0, H1, V0]
#     #     def _accept(q0, p0, q1, p1, H0, H1, V0, V1):
#     #         return q1, p1, 1., [H0, H1, V1]
#     #     def _reject(q0, p0, q1, p1, H0, H1, V0, V1):
#     #         return q0, p0, 0., [H0, H1, V0]
#     #     def _mh(prob, q0, p0, q1, p1, H0, H1, V0, V1):
#     #         prob = jnp.min(1., prob)
#     #         jax.lax.cond(jnp.random.uniform(0., 1., size=1) > prob, _reject, _accept, [q0, p0, q1, p1, H0, H1, V0, V1])
            
#     #     jax.lax.cond(jnp.isnan(prob), _error, _mh, [prob, q0, p0, q1, p1, H0, H1, V0, V1])
    

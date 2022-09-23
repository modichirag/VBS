import numpy as np
import numpy


def fftk(shape, boxsize, symmetric=True, finite=False, dtype=np.float64):
    """ return kvector given a shape (nc, nc, nc) and boxsize 
    """
    k = []
    for d in range(len(shape)):
        kd = numpy.fft.fftfreq(shape[d])
        kd *= 2 * numpy.pi / boxsize * shape[d]
        kdshape = numpy.ones(len(shape), dtype='int')
        if symmetric and d == len(shape) -1:
            kd = kd[:shape[d]//2 + 1]
        kdshape[d] = len(kd)
        kd = kd.reshape(kdshape)

        k.append(kd.astype(dtype))
    del kd, kdshape
    return k


def power(f1, f2=None, boxsize=1.0, k = None, symmetric=True, demean=True, eps=1e-9):
    """
    Calculate power spectrum given density field in real space & boxsize.
    Divide by mean, so mean should be non-zero
    """
    if demean and abs(f1.mean()) < 1e-3:
        print('Add 1 to get nonzero mean of %0.3e'%f1.mean())
        f1 = f1*1 + 1
    if demean and f2 is not None:
        if abs(f2.mean()) < 1e-3:
            print('Add 1 to get nonzero mean of %0.3e'%f2.mean())
            f2 =f2*1 + 1
    
    if symmetric: c1 = numpy.fft.rfftn(f1)
    else: c1 = numpy.fft.fftn(f1)
    if demean : c1 /= c1[0, 0, 0].real
    c1[0, 0, 0] = 0
    if f2 is not None:
        if symmetric: c2 = numpy.fft.rfftn(f2)
        else: c2 = numpy.fft.fftn(f2)
        if demean : c2 /= c2[0, 0, 0].real
        c2[0, 0, 0] = 0
    else:
        c2 = c1
    #x = (c1 * c2.conjugate()).real
    x = c1.real* c2.real + c1.imag*c2.imag
    del c1
    del c2
    if k is None:
        k = fftk(f1.shape, boxsize, symmetric=symmetric)
        k = sum(kk**2 for kk in k)**0.5
    H, edges = numpy.histogram(k.flat, weights=x.flat, bins=f1.shape[0]) 
    N, edges = numpy.histogram(k.flat, bins=edges)
    center= edges[1:] + edges[:-1]
    power = H *boxsize**3 / N
    power[power == 0] = np.NaN
    return 0.5 * center,  power


#################################################################################


def get_ps(iterand, truth, bs):

    ic1, fin1 = iterand
    ic2, fin2 = truth

    pks = []
    #if abs(ic1.mean()) < 1e-3: ic1 += 1
    #if abs(ic.mean()) < 1e-3: ic += 1                                                                                                                  
    k, p1 = power(ic1+1, boxsize=bs)
    k, p2 = power(ic2+1, boxsize=bs)
    k, p12 = power(ic1+1, f2=ic2+1, boxsize=bs)
    pks.append([p1, p2, p12])
    if fin1.mean() < 1e-3: fin1 += 1
    if fin2.mean() < 1e-3: fin2 += 1
    k, p1 = power(fin1, boxsize=bs)
    k, p2 = power(fin2, boxsize=bs)
    k, p12 = power(fin1, f2=fin2, boxsize=bs)
    pks.append([p1, p2, p12])

    return k, pks

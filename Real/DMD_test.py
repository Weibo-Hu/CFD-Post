import matplotlib.pyplot as plt
import numpy as np
import ReductionModel as rm
from timer import timer

#%% define the problem for DMD
def f1(x,t):
    return 1./np.cosh(x+3)*np.exp(2.3j*t)

def f2(x,t):
    return 2./np.cosh(x)*np.tanh(x)*np.exp(2.8j*t)


x = np.linspace(-5, 5, 128)
t = np.linspace(0, 4*np.pi, 256)
tgrid, xgrid = np.meshgrid(t, x)

x1 = f1(xgrid, tgrid)
x2 = f2(xgrid, tgrid)
XX = x1+x2

titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
data   = [x1, x2, XX]
fig = plt.figure(figsize=(17,6))
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
plt.colorbar()
plt.show()

#%% DMD
with timer("DMD test case computing"):
    eigval, phi, coeff= \
        rm.DMD_Standard(XX, t, './', fluc = None)
    
#%% Eigvalue Spectrum
#for eig in eigval:
#    print('Eigenvalue {}: distance from unit circle {}'.format(eig, \
#          np.abs(eig.imag**2+eig.real**2 - 1)))

plt.figure(figsize=(10, 10))
plt.gcf()
ax = plt.gca()
points, = ax.plot(eigval.real, eigval.imag, 'bo', label='eigvalues')
limit = np.max(np.ceil(np.absolute(eigval)))
ax.set_xlim((-limit, limit))
ax.set_ylim((-limit, limit))
plt.xlabel('Real part')
plt.ylabel('Imaginary part')
unit_circle = plt.Circle((0., 0.), 1., color='black', fill=False, \
                         label='unit circle', linestyle='--')
ax.add_artist(unit_circle)
ax.grid(b=True, which = 'both', linestyle = ':')
plt.show()

#%% Modes in space
plt.figure(figsize=(10, 5))
i = 0
modes = phi.T
for i in range(2):
    plt.plot(x, phi[:,i].real)
    plt.title('Modes')
plt.plot(x, f1(x, 0.0)/5.0, ':') # exact results
plt.plot(x, f2(x, 0.0)/-6.0, '--')
plt.show()

#%% Time evolution of each mode
plt.figure(figsize=(10, 5))
dynamics=rm.DMD_Dynamics(eigval, coeff, t)
i = 0
for i in range(2):
    plt.plot(t, dynamics[i,:].real)
    plt.title('Dynamics')
plt.plot(t, f1(0.1, t), ':') # exact results
plt.plot(t, f2(0.1, t), '--')
plt.show()    

#%% reconstruct flow field
reconstruct = rm.DMD_Reconstruct(phi, dynamics)
fig = plt.figure(figsize=(17,6))
for i in range(2):
    n = '13'+str(i+1)
    plt.subplot(int(n))
    phi1 = phi[:,i].reshape(np.size(x), 1)
    dyn1 = dynamics[i,:].reshape(1, np.size(t))
    plt.pcolor(xgrid, tgrid, (phi1@dyn1).real)
    
plt.subplot(133)
plt.pcolor(xgrid, tgrid, reconstruct.real)
plt.show()
#%% abosulte error between exact and reconstruct resutls
err = XX-reconstruct
print("Errors of DMD: ", np.linalg.norm(err))
fig = plt.figure()
plt.pcolor(xgrid, tgrid, (err).real)
fig = plt.colorbar()
plt.show()

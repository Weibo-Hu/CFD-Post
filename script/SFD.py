import numpy as np
import matplotlib.pyplot as plt


DataTab = np.loadtxt("SFD_residual.dat", skiprows=1)
DataTab = DataTab[DataTab[:, 0].argsort()]
time = DataTab[:,0]
L2 = DataTab[:,2]

delta = input ("input filter width: delta =")
chi = input ("input dissipation control coefficient: chi =")

plt.semilogy(time, L2)
# plt.xlim ([0, 3000])
plt.ylim ([1e-10, 1e-1])
plt.xlabel ("time")
plt.ylabel ("residual")
plt.grid(b=True, which='major', color='grey', linestyle='-')
plt.grid(b=True, which='minor', color='grey', linestyle=':', alpha=0.2)
plt.savefig("d"+str(delta)+"x"+str(chi)+".jpg", dpi=600)
plt.show()



# CFD Data Post-process
Post-Processing data using python and tecplot, mainly for CFD results from INCA. In order to run it, you need to install python3, tecplot and pytecplot. This program includes the following features:

* Convert data format from **tecplot** *(.plt/.szplt)* to **python pandas dataframe** *(.h5)*, or the other way around.
* Compute basic flow parameters, like **Reynolds number**, **boundary layer thickness**, **viscosity**, **y+**, **enstrophy**, and so on.
* Dynamic mode decompostion (DMD) analysis
* Proper Orthogonal Decomposition (POD) analysis
* Spectral and statistical analysis

You can follow the code on https://github.com/Weibo-Hu/CFD-Post.git

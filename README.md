Zifeng Li's code for Optics Simulations, also include some Comsol (FEM Software) files,
you can find some classic topics, including **Photonic Crystal & WaveGuids, Fivers, Cavity, Nonlinear Optics, Fourier Optics**

# What you can get?
Diveinto **professional Python Coding Skills** and **useful physics(Optics) simulation skills**, you will know how to do **high performance computation tasks** using powerful tools including **numpy, scipy, pytorch** and useful visualization tool including **matplotlib, seaborn**

# What the code is:
This code base privide some simulations based on fundamental formulars. For example, for Waveguides (fiber is a special kind of waveguids, which has the same Eq but different requirements and targets), we can get helmholtz Eq from Maxwell Eq:

$$
\left[\nabla_T^2 + (\frac{n(x,y)^2\omega^2}{c_0^2} - \beta^2 )\right] U(x,y) = 0
\\
U(x,y,z) = U(x,y) e^{-j\beta z}
$$

In discrete form, it can be turned to a eigenvalue question:

$$
A_{m,n} \phi_n = \lambda \phi_n
$$

which can be solved using scipy.linalg, 

If you divinto this question,you will find that I use scipy.sparse.linalg because of almost 90% lements for the Matrix $A_{m,n}$ is zero, the reason for that is now up for grabs waiting for you.

# How to use:
You are strongly recommended learning and rebuilding this code base,
when you are learning, (after you have learned) Electron Dynamics(电动力学)，and Computational Physica(计算物理)



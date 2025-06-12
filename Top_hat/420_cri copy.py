# -*- coding: utf-8 -*-
#%%
#______________________________________________________________________________________________________________________________________
import numpy as np                          # 用于数组操作
import matplotlib.pyplot as plt             # 绘图
import torch                                 # 使用 PyTorch 进行张量计算
import SLM_1X as slm                         # 包含 SLM 属性、场计算、目标和绘图属性
import CG_1 as cg                           # CG 计算和诊断图
import SV_1 as sv                           # 创建路径和文件夹；保存
import os, shutil                           # 文件夹/文件操作
import time
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
from sipyco.pc_rpc import Client
from scipy.ndimage import binary_dilation, rotate
import torch.nn as nn
from PIL import Image
import CG_2 as cg2                       
import CG_3 as cg3
from Gaussian_2D_Fit import gaussian_2d_fit                      

def centroid(data):
    total = np.sum(data)
    x_c = np.sum(data * x) / total
    y_c = np.sum(data * y) / total
    return x_c, y_c

def gaussian_line2(dim, r0, d,d2 ,sigma, A=1.0, save_param=False, device='cuda'):
    """
    Create n x n target: 
    Gaussian line centered on r0 = (x0, y0) with length 'd', width
    'sigma', and amplitude 'A'.
    
    Args:
        n (int): Size of the grid (n x n).
        r0 (list/tuple): Center of the Gaussian line [x0, y0].
        d (float): Length of the Gaussian line.
        sigma (float): Width of the Gaussian line.
        A (float): Amplitude of the Gaussian line.
        save_param (bool): Whether to return parameters used.
        device (str): Device to store tensors, e.g., 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: 2D Gaussian line.
        str (optional): Parameters used (if save_param is True).
    """
    # Initialization
    cols, rows = dim

    # Initialization
    device = 'cuda'
    x = torch.arange(rows, dtype=torch.float32, device=device)  # Grid points along cols
    y = torch.arange(cols, dtype=torch.float32, device=device)  # Grid points along rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Cartesian indexing    
    # Target definition
    fx = 0.5 * (torch.abs(X - d / 2. - r0[0]) + torch.abs(X + d / 2. - r0[0]) - d)
    fy = 0.5 * (torch.abs(Y - d2 / 2. - r0[1]) + torch.abs(Y + d2 / 2. - r0[1]) - d2)
    z = A * torch.exp(-(fx**2 + fy**2) / sigma**2)

    if save_param:
        param_used = f"gaussian_line | n={rows} | r0={r0} | d={d} | sigma={sigma} | A={A}"
        return z, param_used
    else:
        return z
    
def get_centre_range(n,N):
    # returns the indices to use given an nxn SLM
    return int(N/2)-int(n/2),int(N/2)+int(n/2)

class InverseFourierOp:
    def __init__(self):
        pass

    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi  # Return the input tensors for now
    
    def perform(self, node, inputs, output_storage):
        xr, xi = inputs  # 保持输入方式不变
        x = xr + 1j * xi
        nx, ny = xr.shape
        s = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x))) * (nx * ny)
        output_storage[0][0] = s.real
        output_storage[1][0] = s.imag

class FourierOp:
    __props__ = ()
    
    def make_node(self, xr, xi):
        xr = torch.as_tensor(xr)
        xi = torch.as_tensor(xi)
        return xr, xi  # 返回输入张量

    def perform(self, node, inputs, output_storage):
        x = inputs[0] + 1j * inputs[1]
        s = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
        output_storage[0] = s.real
        output_storage[1] = s.imag
        z_r = output_storage[0]
        z_i = output_storage[1]
        
    def grad(self, inputs, output_gradients):
        z_r = output_gradients[0]
        z_i = output_gradients[1]
        y = InverseFourierOp()(z_r, z_i)
        return y

    def __call__(self, xr, xi):
        # 创建节点
        inputs = self.make_node(xr, xi)
        output_storage = [torch.empty_like(xr), torch.empty_like(xi)]
        # 执行傅里叶变换
        self.perform(None, inputs, output_storage)
        return output_storage[0], output_storage[1]  # 返回实部和虚部

fft = FourierOp()

device = torch.device("cuda")
torch.set_default_device(device)

def phase_guess(dim, D, asp, R, ang, B, save_param=False):
    """
    Create n x n guess phase: 
    'D' required radius of shift from origin
    'asp' aspect ratio of "spreading" for quadratic profile
    'R' required curvature of quadratic profile
    'ang' required angle of shift from origin
    'B' radius of ring in output plane
    """
    cols, rows = dim
    
    # Initialization
    x = torch.arange(rows) - rows / 2  # Columns
    y = torch.arange(cols) - cols / 2  # Rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
    z = torch.zeros(size=(rows, cols))

    # target definition
    KL = D*((X/shr)*torch.cos(ang)+(Y/shr)*torch.sin(ang));
    KQ = 3*R*((asp*(torch.pow((X/shr),2))+(1-asp)*(torch.pow((Y/shr),2))));
    KC = B*torch.pow((torch.pow((X/shr),2)+torch.pow((Y/shr),2)),0.5);
    z = KC+KQ+KL;
    z = torch.reshape(z, (rows * cols,))
    
    if save_param :
        param_used = "phase_guess | n={0} | D={1} | asp={2} | R={3} | ang={4} | B={5}".format(rows, D, asp, R, ang, B)
        return z, param_used
    else :
        return z
    
def gaussian_linex(dim, r0, d,d2, sigma, A=1.0, save_param=False, device='cuda'):

    cols, rows = dim

    # Initialization
    device = 'cuda'
    x = torch.arange(rows, dtype=torch.float64, device=device)  # Grid points along cols
    y = torch.arange(cols, dtype=torch.float64, device=device)  # Grid points along rows
    X, Y = torch.meshgrid(x, y, indexing='xy')  # Cartesian indexing    
    # Target definition
    fx = 0.5 * (torch.abs(X - d / 2. - r0[1]) + torch.abs(X + d / 2. - r0[1]) - d)
    fy= 0.5*(torch.abs(Y-d2/2.-r0[0])+torch.abs(Y+d2/2.-r0[0])-d2)
    z = A * torch.exp(-(fx**2 + fy**2) / sigma**2)

    if save_param:
        param_used = f"gaussian_line | n={rows} | r0={r0} | d={d} | sigma={sigma} | A={A}"
        return z, param_used
    else:
        return z
    









#%%
N = [1024,1272] # SLM 是 NxN 像素
shr=1024/256
#   ================================================================================================
#   |          SLM 像素
#   ================================================================================================
numb=2
# NT =  [x * numb for x in N]  # 模型输出平面为 NTxNT 像素阵列 - 更高分辨率
# NT =  [2514,2514]
# NT =  [4096,4096]
# NT =  [7540,7540]  # 模型输出平面为 NTxNT 像素阵列 - 更高分辨率
NT=[6000,6000]
# NT=[4800,4800]

#   ================================================================================================
#   |          激光束
#   ================================================================================================
spix = 0.0125  # SLM 像素大小，单位为 mm
lambda_ = 420e-9
magnification=1
focallength=0.25
focal_spy=lambda_*focallength/(NT[0]*(spix/1000)*magnification)
focal_spx=lambda_*focallength/(NT[1]*(spix/1000)*magnification)
sx = 1.1  # x 轴强度束大小，单位为 mm (1/e^2)
sy = 1.1# y 轴强度束大小，单位为 mm (1/e^2)









#%%
#   ===   激光幅度   ===============================================
L, Lp = slm.laser_gaussian(dim=N, r0=(0, 0), sigmax=sx/spix, sigmay=sy/spix, A=1.0, save_param=True)

# 用于将激光强度的总和匹配到目标强度的总和的归一化
# ===  激光归一化 | 请勿删除  ================
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
L = L * torch.pow(10000 / I_L_tot, 0.5)                      #
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
# ===  激光归一化 | 请勿删除  ================

#   ================================================================================================
#   |          目标幅度、目标相位、加权 cg、加权 i
#   ================================================================================================
# param = [1., round(150e-6/focal_spx), round(25e-6/focal_spx), NT/2., NT/2., 9]  # [d2, sigma, l, roi, roj, C1]
param = [torch.tensor(1.), torch.tensor(round(100e-6/focal_spx)), torch.tensor(round(50e-6/focal_spy)), torch.tensor(round(146e-6/focal_spx)+NT[0]/2.), torch.tensor(round(31e-6/focal_spx)+NT[1]/2.), torch.tensor(9)]  # [d2, sigma, l, roi, roj, C1]

d2 = param[0]  # 加权区域宽度
sigma = param[1]  # Laguerre 高斯宽度
l = param[2]
r0i = param[3]
r0j = param[4]
r0 = torch.tensor([r0i, r0j], dtype=torch.float64)   # 模式位置
C1 = param[5]  # 陡度因子

#   ===   目标幅度   ==============================================
# Ta, Tap = slm.target_lg(n=NT, r0=r0, w=sigma, l=l, A=1.0, save_param=True)
# Ta, Tap = slm.gaussian_line(dim=NT, r0=r0, d=sigma,sigma=l, A=1.0, save_param=True)
# Ta, Tap = gaussian_linex(dim=NT, r0=r0, d2=2,d=sigma,sigma=l, A=1.0, save_param=True)

Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=round(0e-6/focal_spx),sigma=round(60e-6/focal_spx),A=1.0, save_param=True)
#   ===   目标相位   ==================================================
# P, Pp = slm.phase_spinning_continuous(n=NT, r0=r0, save_param=True)

# P, Pp =slm.gaussian_line_phase(n=NT, r0=r0, d=l,sigma=sigma, save_param=True)
# P, Pp = slm.phase_inverse_square(n=NT[0], r0=r0, save_param=True)
# P, Pp = phase_qiu(n=NT[0], beta=7.479982508547126e-06/(1e-1),r0=r0, save_param=True)
P, Pp = slm.phase_flat(dim=NT,v=0, save_param=True)

#   ===   加权 cg   ==================================================
# Weighting, Weightingp = slm.gaussian_top_round(dim=NT, r0=r0, d=2+d2 + sigma, sigma=2, A=1.0, save_param=True)
Weighting, Weightingp =slm.flat_top_round(dim=NT, r0=r0,d=round(190e-6/focal_spx),A=1.0, save_param=True)
Wcg, Wcgp = slm.weighting_value(M=Weighting, p=1E-4, v=0, save_param=True)

# 用于将激光强度的总和匹配到目标强度的总和的归一化
# ===  目标归一化 | 请勿删除  ================
Ta = Ta * Wcg                                              #
P = P * Wcg                                                #
I_Ta_w = torch.sum(torch.pow(Ta, 2.))                           #
Ta = Ta * torch.pow(I_L_tot / (I_Ta_w), 0.5)                #
I_Ta = torch.pow(torch.abs(Ta), 2.)                             #
# ===  目标归一化 | 请勿删除  ================

#   |          初始 SLM 相位
curv = 3.7  # 猜测相位的曲率，单位为 mrad px^-2 (R)
# init_phi, ipp = slm.phase_guess(N, -torch.pi/2.0, 0.5, curv/1000, torch.pi/4, 0, save_param=True)  # 猜测相位
# init_phi, ipp = phase_guess(N, -0.3*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/3.7), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
# init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.1, 5.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
# init_phi, ipp = phase_guess(N, -0.15*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.0), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
init_phi, ipp = phase_guess(N, 0.09*torch.pi, 0.5, 2.0/1000, torch.tensor(0.15*torch.pi), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位

# init_phi=torch.random.uniform(0, 2 * torch.pi, (256, 256))
# init_phi=torch.reshape(init_phi,256**2)
# init_phi, ipp = phase_guess(N, -0.16*torch.pi, 0.9, 3/1000, torch.tensor(torch.pi/3.7), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位

# 计算与此相位猜测相关的电场和输出平面强度：
init_phi2 = torch.reshape(init_phi, [N[0],N[1]])
real_part = L * torch.cos(init_phi2) 
imag_part = L * torch.sin(init_phi2)
E_guess = torch.complex(real_part, imag_part) 
E_guess_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(E_guess)))
I_guess_out_amp = torch.pow(torch.abs(E_guess_out), 2)
phase_guess_out = torch.angle(E_guess_out)

#   ===   SLM 对象   ====================================================
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)


#   ================================================================================================
#   |          成本函数
#   ================================================================================================
# E_out_amp_tensor = torch.tensor(slm_opt.E_out_amp, dtype=torch.float32)
E_out_amp_tensor = slm_opt.E_out_amp # 使用 clone().detach()


# E_out_p_tensor = torch.tensor(slm_opt.E_out_p, dtype=torch.float32)
      # 使用 clone().detach()

# overlap = torch.sum(Ta_tensor * E_out_amp_tensor * Wcg_tensor * torch.cos(slm_opt.E_out_p - P_tensor))
# overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta_tensor, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp_tensor * Wcg_tensor, 2)), 0.5))
# cost_SE = torch.pow(10, C1_tensor) * torch.pow((1 - overlap), 2)
def cost_SE(phi):
    slm_opt.phi.data = torch.tensor(phi,dtype=torch.float64)# 更新参数
    slm_opt.update_output(NT)
    E_out_a=slm_opt.E_out_amp
    E_out_ph=slm_opt.E_out_p
    # overlap = torch.sum(Ta * E_out_a * Wcg * torch.cos(E_out_ph - P))
    # overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_a * Wcg, 2)), 0.5))
    # cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
    cost=torch.pow(10, C1)*torch.sum(torch.pow((Ta**2 - E_out_a**2), 2))
    return cost

# def cost_SE2(phi):
#     slm_opt.phi.data = torch.tensor(phi,dtype=torch.float64)# 更新参数
#     slm_opt.update_output(NT)
#     # E_out_a=slm_opt.E_out_amp
#     E_out_I = slm_opt.E_out_2
#     # E_out_ph=slm_opt.E_out_p
#     # overlap = torch.sum(Ta_tensor * E_out_a * Wcg_tensor * torch.cos(E_out_ph - P_tensor))
#     # overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta_tensor, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_a * Wcg_tensor, 2)), 0.5))
#     # cost = torch.pow(10, C1_tensor) * torch.pow((1 - overlap), 2)
#     cost=torch.sum(torch.pow((Ta**2 - E_out_I), 2))
#     return cost
# # cost1=cost_SE(init_phi).item()
# # cost2=cost_SE2(init_phi).item()
# # alpha=0.9
# def cost_SE3(phi):
#     slm_opt.phi.data = torch.tensor(phi,dtype=torch.float64)# 更新参数
#     slm_opt.update_output(NT)
#     E_out_a=slm_opt.E_out_amp
#     E_out_I = slm_opt.E_out_2
#     E_out_ph=slm_opt.E_out_p
#     overlap = torch.sum(Ta * E_out_a * Wcg * torch.cos(E_out_ph - P))
#     overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_a * Wcg, 2)), 0.5))
#     cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)+1000*torch.sum(torch.pow((Ta**2 - E_out_I), 2))

#     return cost

profile_s=L
if not torch.is_complex(profile_s):
    profile_s = profile_s.to(torch.complex128)
n_pixelsx = int(N[0])
n_pixelsy = int(N[1]) 
profile_s_r = profile_s.real.type(torch.float64)
profile_s_i = profile_s.imag.type(torch.float64)
A0 = 1. / np.sqrt(NT[0] * NT[1])  # Linked to the fourier transform. Keeps the same quantity of light between the input and the output


def cost_SE_gpu(phi):
    zero_matrix = torch.zeros((NT[0], NT[1]), dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)
    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
    # cost=torch.pow(10, C1)*torch.sum(torch.pow((Ta**2 - E_out_amp**2), 2))

    return cost
#   ================================================================================================
#   |          绘图
#   ================================================================================================
# p1 = [torch.pow(torch.abs(Ta), 2).cpu(), P.cpu(), Wcg.cpu(), phase_guess_out.cpu(), I_guess_out_amp.cpu()]  # 数据
# sz1 = [[], [], [], [], []]  # 缩放轴
# t1 = ['ta', 'tp', 'wcg', 'gp', 'ga']  # 标题
# v1 = [[], [], [], [], []]  # 限制
# c1 = [[], [], [], [], []]  # 颜色
# slm.n_plot(p=p1, t=t1, v=v1, c=c1)
# plt.show() # 当图形打开时，必须关闭以继续程序
# W20 = slm.weighting_value(M=I_Ta, p=0.8, v=0, save_param=False) # 前10%的加权
#   ================================================================================================
#   |          共轭梯度
#   ================================================================================================
nb_iter = 3000




#%%检查参数
plt.imshow(I_Ta.detach().cpu()[2930:3130,3000:3200])
plt.show()

mu = np.arctan(31/146)
D = 2*np.pi*12.5/0.42/250e3*146/np.cos(mu) *4

init_phi, ipp = phase_guess(N, D, 0.5, 4.2/1000, torch.tensor(mu), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)
plt.imshow(slm_opt.E_out_2.detach().cpu()[2930:3130,3000:3200])
plt.show()
data=slm_opt.E_out_2.detach().cpu()
# print(f"质心法中心: ({centroid(data)[0]:.2f}, {centroid(data)[1]:.2f})")
popt,_,_ = gaussian_2d_fit(I_Ta.detach().cpu()[2930:3130,3000:3200],show=True)
print(popt)

popt,_,_ = gaussian_2d_fit(slm_opt.E_out_2.detach().cpu()[2930:3130,3000:3200],show=True)
print(popt)




#%%
fft = FourierOp()
cg1 = cg2.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi, ipp),
            nb_iter=nb_iter,
            numb=numb,
            slm_opt=slm_opt,
            cost_SE=cost_SE_gpu,
            show=True,
            lr=0.1,
            goal=0.1)



#%%
np.save('420_gaussian_tophat_phase_f=250mm_size=(25_40)_shift=(147_31)',cg1.slm_phase_end)


#%%
base=I_Ta
#%%
for i in range(3):
    I_out=(cg1.I_out*cg1.Wcg).detach()
    I_out=I_out*I_L_tot/(I_out.sum())
    I=I_Ta+0.25*(base-I_out)
    Ta=torch.pow(I,0.5)
    I_Ta=torch.pow(Ta,2)
    plt.imshow(I_Ta.detach().cpu()[2950:3200,2950:3200])
    plt.show()

    init_phi, ipp = phase_guess(N, 0.085*torch.pi, 0.5, 3.5/1000, torch.tensor(0.29*torch.pi), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
    slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)
    plt.imshow(slm_opt.E_out_2.detach().cpu()[2950:3200,2950:3200])
    plt.show()
    def cost_SE_gpu(phi):
        zero_matrix = torch.zeros((NT[0], NT[1]), dtype=torch.float64)

        # Phi and its momentum for use in gradient descent with momentum:
        phi_reshaped = phi.view(n_pixelsx, n_pixelsy)
        # E_in (n_pixels**2):
        S_r = torch.tensor(profile_s_r, dtype=torch.float64)
        S_i = torch.tensor(profile_s_i, dtype=torch.float64)
        E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
        E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

        # 填充输入场
        idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
        idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

        E_in_r_pad = zero_matrix.clone()
        E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

        E_in_i_pad = zero_matrix.clone()
        E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

        phi_padded = zero_matrix.clone()
        phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

        # 计算输出场（傅里叶变换）
        E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

        # 计算输出强度
        E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

        # 计算输出相位
        E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位

        # 输出振幅
        E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
        overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
        overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
        cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
        # cost=torch.pow(10, C1)*torch.sum(torch.pow((Ta**2 - E_out_amp**2), 2))

        return cost
    fft = FourierOp()
    cg1 = cg2.CG(L_Lp=(L, Lp),
                r0=r0,
                Ta_Tap=(Ta, Tap),
                P_Pp=(P, Pp),
                Wcg_Wcgp=(Wcg, Wcgp),
                init_phi_ipp=(init_phi, ipp),
                nb_iter=nb_iter,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu,
                show=True,
                lr=0.1,
                goal=0.01)
#%%
np.save('420_gaussian_tophat_phase_f=250mm_size=(30_60)_shift=(120_42)',cg1.slm_phase_end)
















#%%

# 如果 'show' 为 True，将显示诊断图
# 注意：图形需要关闭，代码才能继续
phi_after=cg1.res[0]
init_phi4=torch.from_numpy(np.load("phase_after_amp.npy")).to('cuda')
init_phi3=(torch.from_numpy(cg1.slm_phase_end)-cg1.slm_phase_init).to('cuda')
#%%
cg1 = cg.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi, ipp),
            nb_iter=nb_iter,
            slm_opt=slm_opt,
            cost_SE=cost_SE2,
            show=True)
# 如果 'show' 为 True，将显示诊断图
# 注意：图形需要关闭，代码才能继续
phi_after=cg1.res[0]

#%%
#   ================================================================================================
#   |          保存
#   ================================================================================================
sv1 = sv.SV(cg1=cg1,
            name_dir=os.getcwd(),
            name_folder="flat_top_light_813_c",
            save=True,
            save_info=True,
            save_I=True,
            save_P=True,
            save_slm=True,
            save_weighting=True,
            save_figs=True,
            visua_iter=False,
            err_iter=False)


#%%
slm_opt_end = slm.SLM(NT=NT,N=N, initial_phi=phi_after, profile_s=L)
E_in_r_pad=slm_opt_end.E_in_r_pad
E_in_i_pad=slm_opt_end.E_in_i_pad
E_out_r, E_out_i = (fft(E_in_r_pad, E_in_i_pad))
# Output intensity:
E_out_2 = E_out_r ** 2 + E_out_i ** 2

# E_out_phi:
E_out_p = torch.atan2(E_out_i, E_out_r)

# Output amplitude:
E_out_amp = torch.sqrt(E_out_2)
plt.imshow(E_out_2.detach().numpy())
plt.colorbar()
plt.show()

# %%
i, j = 171, 171  # 选择图像中心点

# 定义不同的传播距离 z 取值
z_values = torch.linspace(-0.002, 0.002, 200)  # 传播距离范围，从 0 到 2cm

# 用于存储每个传播距离下的光强值
intensity_curve = []

# 计算每个 z 平面上的光场强度
for z in z_values:
    # 计算传播因子
    K = torch.sqrt((2 * torch.pi / lambda_)**2 * (KX**2 + KY**2))  # 空间频率的传播因子
    prop_factor = torch.exp(1j * z * K)  # 传播因子

    # 将传播因子应用到频谱上
    E_out_r_z = E_out_r * prop_factor.real - E_out_i * prop_factor.imag
    E_out_i_z = E_out_r * prop_factor.imag + E_out_i * prop_factor.real

    # 计算该点的光强：取出 (i, j) 位置的光场强度
    E_out_2_z = E_out_r_z ** 2 + E_out_i_z ** 2  # 光强
    intensity_curve.append(E_out_2_z[i, j].detach().numpy())  # 存储该点的光强

# 绘制光强随传播距离 z 的变化曲线
plt.plot(z_values, intensity_curve)
plt.xlabel('Propagation Distance (z) [m]')
plt.ylabel('Intensity at point (i, j)')
plt.title('Intensity Curve at point (i, j) for different z values')
plt.grid(True)
plt.show()
# %%
def fresnel_lens_phase_generate(shift_distance, SLMRes=(1024,1272), x0=N[0]/2, y0=N[1]/2, pixelpitch=12.5,wavelength=0.420,focallength=250000,magnification=1):
        '''
        the fresnel lens phase, see notion for more details.
        '''

        Xps, Yps = torch.meshgrid(torch.linspace(0, SLMRes[0], SLMRes[0]), torch.linspace(0, SLMRes[1], SLMRes[1]))
        Xps = Xps.to(device)
        Yps = Yps.to(device)

        X = (Xps-x0)*pixelpitch
        Y = (Yps-y0)*pixelpitch

        fresnel_lens_phase = torch.fmod(torch.pi*(X**2+Y**2)*shift_distance/(wavelength*focallength**2)*magnification**2,2*torch.pi)

        return torch.remainder(fresnel_lens_phase,2*torch.pi).to('cuda')
#%%
def phase_to_fftField_3d(SLM_Phase,fresnel_lens):
        
    SLM_Field = torch.multiply(L, torch.exp(1j*SLM_Phase))
    SLM_Field_shift = torch.fft.fftshift(SLM_Field*torch.exp(1j*(fresnel_lens_phase_generate(-fresnel_lens))))
    fftSLM = torch.fft.fft2(SLM_Field_shift)
    fftSLMShift = torch.fft.fftshift(fftSLM)
    fftSLM_norm = torch.sqrt(torch.sum(torch.square(torch.abs(fftSLMShift))))
    fftSLMShift_norm = fftSLMShift/fftSLM_norm

    fftAmp = torch.abs(fftSLMShift_norm)
    fftPhase = torch.angle(fftSLMShift_norm)
    return fftAmp.cpu(), fftPhase.cpu()
# %%
fresnel_lens_phase_generate(-10)
# %%
#%%
i, j = 1024, 1024
z=0
phase=cg2.slm_phase_end
padded_matrix = torch.pad(phase, pad_width=384, mode='constant', constant_values=0)
slm_opt_end = slm.SLM(NT=NT, initial_phi=cg2.res[0]+fresnel_lens_phase_generate(z), profile_s=L)
E_in_r_pad=slm_opt_end.E_in_r_pad
E_in_i_pad=slm_opt_end.E_in_i_pad
E_out_r, E_out_i = (fft(E_in_r_pad, E_in_i_pad))
# Output intensity:
E_out_2 = E_out_r ** 2 + E_out_i ** 2

# E_out_phi:
E_out_p = torch.atan2(E_out_i, E_out_r)
k=E_out_2.detach().numpy()
# Output amplitude:
plt.imshow(k)
plt.colorbar()
plt.show()
wei=torch.zeros_like(E_out_2.detach().numpy())
wei[torch.where(I_Ta>0.1*I_Ta.max())]=1
s=E_out_2.detach().numpy()*wei
print(torch.where(s==s.max()))
print(E_out_2[i,j])
plt.imshow(k[950:1100,950:1100])

# %%
i, j = 1024, 1024  # 选择图像中心点

# 定义不同的传播距离 z 取值
z_values = torch.linspace(-100, 100, 1)  # 传播距离范围，从 0 到 2cm

# 用于存储每个传播距离下的光强值
intensity_curve = []

# 计算每个 z 平面上的光场强度
for z in z_values:
    slm_opt_end = slm.SLM(NT=NT, initial_phi=cg2.res[0]+fresnel_lens_phase_generate(z), profile_s=L)
    E_in_r_pad=slm_opt_end.E_in_r_pad
    E_in_i_pad=slm_opt_end.E_in_i_pad
    E_out_r, E_out_i = (fft(E_in_r_pad, E_in_i_pad))
    # Output intensity:
    E_out_2 = E_out_r ** 2 + E_out_i ** 2

    intensity_curve.append(E_out_2[i, j].detach().numpy())  # 存储该点的光强

# 绘制光强随传播距离 z 的变化曲线
plt.plot(z_values, intensity_curve)
plt.xlabel('Propagation Distance (z) [um]')
plt.ylabel('Intensity at point (i, j)')
plt.title('Intensity Curve at point (i, j) for different z values')
plt.grid(True)
plt.show()
# %%



# %%

#%% camera connection, set and initialize

IDS_Camera = Client('192.168.31.15','11000','IDS_Camera')


#%%
IDS_Camera.SetROI()
IDS_Camera.SetBitDepth(12)

#%%
exposure_time = 200
IDS_Camera.SetExposureTime(exposure_time)

# %%
IDS_Camera.PrepareAcquisition()
IDS_Camera.AllocAndAnnounceBuffers()
IDS_Camera.StartAcquisition()

#%%
image = IDS_Camera.GetImage()

# %%
plt.figure(figsize=(20,20))
plt.imshow(image)
plt.colorbar()
print('max:',image[1315:1360,1248:1352].max())



# %%
IDS_Camera.StopAcquisition()
IDS_Camera.Close()
#%%
# image=np.load('C:\\Users\\Su Shi\\Desktop\\ftl\\after1.png')
imagex=np.array(Image.open('C:\\Users\\Su Shi\\Desktop\\ftl\\100.png'))
image = rotate(imagex,angle=0,reshape=False)
image.astype(np.float32)
# image=np.load('60um_img1.npy')
# testimg=image
# testimg[1315:1360,center[1]]=0
# plt.imshow(testimg[1315:1360,1248:1352])
#%%
ids_spa=3.45#um
center=[1015,1145]
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (sigma**2))
# range=[1000:1025,1100:1200]

#%%
list = np.arange(-2, 3, 1)
x=range(100)
result=[]
for i in list:
    data=image[center[0]+i,1100:1200]
    result.append(data.mean())
    # popt, pcov = curve_fit(gaussian, x, data, p0=[3500, 20, 20])  # 初始猜测值为 [A, mu, sigma]
    # fitted_A, fitted_mu, fitted_sigma = popt
    # result.append(fitted_mu)
    # print(fitted_mu)
    plt.plot(data)
    # plt.plot(x, gaussian(x, *popt), label='拟合曲线', color='red', linewidth=2)
#%%
list = np.arange(-20, 21, 1)
x=range(25)
result=[]
for i in list:
    data=image[1000:1025,center[1]+i]
    popt, pcov = curve_fit(gaussian, x, data, p0=[3500, 20, 20])  # 初始猜测值为 [A, mu, sigma]
    fitted_A, fitted_mu, fitted_sigma = popt
    result.append(fitted_mu.item())
    print(fitted_mu)
    plt.plot(data)
    plt.plot(x, gaussian(x, *popt), label='拟合曲线', color='red', linewidth=2)
#%%
data=image[1013,1098:1194]
print(data.std()/data.mean())
plt.plot(data)
#%%
cutted_image = np.flip(image[1000:1032,1098:1200],0)
# cutted_image = image[1002:1022,1100:1192]

# rotated_image = rotate(cutted_image,angle=-4.8,reshape=False)
# rotated_image = rotate(cutted_image,angle=-0.2,reshape=False)

plt.figure(figsize=(20,20))
plt.imshow(cutted_image)
plt.colorbar()
plt.show()

# plt.figure(figsize=(20,20))
# plt.imshow(rotated_image)
# plt.colorbar()
# plt.show()
bin_size=2
img_resized=cutted_image.reshape(cutted_image.shape[0]//bin_size,bin_size,
                                 cutted_image.shape[1]//bin_size,bin_size).mean(axis=(1,3))
y_numb=cutted_image.shape[0]//bin_size
x_numb=cutted_image.shape[1]//bin_size
path1='Ta_0f.npy'
path2='Ta_3f_b.npy'
Ta=torch.from_numpy(np.load(path1))
last_Ta=torch.from_numpy(np.load(path2))

last_Ta_cut=last_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2),int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)]
last_I_Ta=torch.pow(torch.abs(last_Ta_cut), 2.)

Ta_cut=Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2),int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)]
I_Ta=torch.pow(torch.abs(Ta_cut), 2.)
img_nomlz = img_resized * (I_Ta.sum().item() / img_resized.sum())
plt.figure(figsize=(20,20))
plt.imshow(img_nomlz)
plt.colorbar()
plt.show()
#%%
cost=I_Ta/img_nomlz
# cost=I_Ta/img_nomlz*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
I_new=last_I_Ta*torch.pow(cost,0.5)
I_new_nomlz=I_new * (I_Ta.sum().item() / I_new.sum())
new_Tax=torch.pow(I_new_nomlz,0.5)
new_Ta=last_Ta
new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2),int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)]=new_Tax
print((new_Ta**2).sum().item())
#%%
cutted_image = np.flip(image[1000:1032,1090:1206],0)

rows, cols = cutted_image.shape

# 创建 X 和 Y 坐标
x = np.arange(cols)  # 列索引为 X 坐标
y = np.arange(rows)  # 行索引为 Y 坐标
X, Y = np.meshgrid(x, y)

# Z 值直接是你的矩阵
Z = cutted_image

# 创建 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# 设置标题和轴标签
ax.set_title('3D Visualization of rotated_image')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()



# %%
y_numb=round(cutted_image.shape[0]*ids_spa/(focal_spy*10**6))
x_numb=round(cutted_image.shape[1]*ids_spa/(focal_spx*10**6))
path1='Ta_adapt0_c.npy'
path2='Ta_adapt1_d.npy'
Ta=torch.from_numpy(np.load(path1))
last_Ta=torch.from_numpy(np.load(path2))

last_Ta_cut=last_Ta[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
last_I_Ta=torch.pow(torch.abs(last_Ta_cut), 2.)

Ta_cut=Ta[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
img_resized=zoom(cutted_image, (Ta_cut.shape[0]/cutted_image.shape[0], Ta_cut.shape[1]/cutted_image.shape[1]), order=3)
I_Ta=torch.pow(torch.abs(Ta_cut), 2.)
img_nomlz = img_resized * (I_Ta.sum().item() / img_resized.sum())
#%%
cost=(I_Ta-img_nomlz)
# cost=(I_Ta-img_nomlz)*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
new_goal=last_I_Ta+0.1*cost
new_goal = torch.clamp(new_goal, min=0)
new_goal=new_goal*I_Ta.sum().item()/new_goal.sum().item()
new_Tax=torch.pow(new_goal,0.5)
new_Ta=last_Ta
new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2),int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)]=new_Tax
# new_Ta=new_Tay*W20+(1-W20)*torch.from_numpy(np.load(path))
# new_Ta=new_Ta*torch.pow(I_L_tot.cpu() / (new_Ta**2).sum(), 0.5)
print((new_Ta**2).sum().item())


#%%
ta_nomlz=np.pow(img_nomlz,0.5)
cost=Ta_cut-ta_nomlz
new_goal=Ta_cut+1*cost
new_goal = torch.clamp(new_goal, min=0)
new_Tax=new_goal*torch.pow(I_Ta.sum()/(torch.pow(torch.abs(new_goal), 2.).sum()),0.5)
new_Ta=Ta
new_Ta[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]=new_Tax
#%%
np.save('Ta_4f_b.npy',new_Ta.cpu())
#%%
new_Ta=torch.from_numpy(np.load('Ta_4f.npy'))
#%%
# cols, rows = N
# new_init_phi=torch.reshape(torch.from_numpy(np.load('813-a-1.npy')),(rows * cols,))
init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
fft = FourierOp()
new_Ta=new_Ta.to('cuda')
def cost_SE_gpu_x(phi):
    zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
    zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

    # Phi and its momentum for use in gradient descent with momentum:
    phi_rate = torch.zeros_like(phi, dtype=torch.float64)
    phi_reshaped = phi.view(n_pixelsx, n_pixelsy)

    # E_in (n_pixels**2):
    S_r = torch.tensor(profile_s_r, dtype=torch.float64)
    S_i = torch.tensor(profile_s_i, dtype=torch.float64)
    E_in_r = A0 * (S_r * torch.cos(phi_reshaped) - S_i * torch.sin(phi_reshaped))
    E_in_i = A0 * (S_i * torch.cos(phi_reshaped) + S_r * torch.sin(phi_reshaped))

    # 填充输入场
    idx_0x, idx_1x = get_centre_range(n_pixelsx,NT[0])
    idx_0y, idx_1y = get_centre_range(n_pixelsy,NT[1])

    E_in_r_pad = zero_matrix.clone()
    E_in_r_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_r  # 填充实际部分

    E_in_i_pad = zero_matrix.clone()
    E_in_i_pad[idx_0x:idx_1x, idx_0y:idx_1y] = E_in_i  # 填充虚部

    phi_padded = zero_matrix.clone()
    phi_padded[idx_0x:idx_1x, idx_0y:idx_1y] = phi_reshaped

    # 计算输出场（傅里叶变换）
    E_out_r, E_out_i = fft(E_in_r_pad, E_in_i_pad)  # 使用FFT计算输出场的实部和虚部

    # 计算输出强度
    E_out_2 = E_out_r ** 2 + E_out_i ** 2  # 输出强度为实部和虚部的平方和

    # 计算输出相位
    E_out_p = torch.atan2(E_out_i, E_out_r)  # 输出相位
    E_out_p_nopad = E_out_p[idx_0x:idx_1x, idx_0y:idx_1y]  # 取出有效区域的相位

    # 输出振幅
    E_out_amp = torch.sqrt(E_out_2)  # 输出振幅
    overlap = torch.sum(new_Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(new_Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
    return cost
cg1 = cg2.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(new_Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi, ipp),
            nb_iter=50,
            numb=numb,
            slm_opt=slm_opt,
            cost_SE=cost_SE_gpu_x,
            show=True,
            goal=2)
#%%
np.save('813-f-4b.npy',cg1.slm_phase_end)

#%%
# 相机反馈
init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
new_Ta=new_Ta.to('cuda')
def cost_SE_x(phi):
    slm_opt.phi.data = torch.tensor(phi,dtype=torch.float64)# 更新参数
    slm_opt.update_output(NT)
    E_out_a=slm_opt.E_out_amp
    E_out_I = slm_opt.E_out_2
    E_out_ph=slm_opt.E_out_p
    overlap = torch.sum(new_Ta * E_out_a * Wcg * torch.cos(E_out_ph - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(new_Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_a * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
    return cost
cg1 = cg.CG(L_Lp=(L, Lp),
            r0=r0,
            Ta_Tap=(new_Ta, Tap),
            P_Pp=(P, Pp),
            Wcg_Wcgp=(Wcg, Wcgp),
            init_phi_ipp=(init_phi, ipp),
            nb_iter=200,
            numb=numb,
            slm_opt=slm_opt,
            cost_SE=cost_SE_x,
            show=True)
# %%
import numpy as np
import scipy as sp
from scipy import fft
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import torch
import IMGpy
import slmpy
# %%
for i in range(5):
    path=f'813-f-{i}b.npy'
    tophatscreen=255*np.load(path)/(2*np.pi)
    # tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
    tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
    time.sleep(5)
    slmx = slmpy.SLMdisplay(monitor=1,isImageLock = True)
    slmx.updateArray(tophat_screen_Corrected)
#%%
path='813-f-3b.npy'
tophatscreen=np.mod((255*np.load(path)/(2*np.pi)+255*np.array(fresnel_lens_phase_generate(1000))/(2*np.pi)),256)
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx = slmpy.SLMdisplay(monitor=1,isImageLock = True)
slmx.updateArray(tophat_screen_Corrected)
# %%
slmx.close()
# %%
init_phi, ipp = phase_guess(N, -0.3*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/3.7), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)
plt.imshow(slm_opt.E_out_2.detach().cpu()[1720:1920,1720:1920])

plt.imshow(I_Ta.cpu()[1720:1920,1720:1920])
# %%
freeze_mask = torch.ones_like(phi, dtype=torch.bool).to('cuda')  # 所有元素初始参与优化
freeze_threshold=0.0001
for step in range(nb_iter):
    # 仅将未冻结的参数传递给优化器
    unfrozen_phi = phi[freeze_mask].clone().detach().requires_grad_(True)
    optimizer = torch.optim.LBFGS([unfrozen_phi], lr=0.01)
    
    def closure():
        optimizer.zero_grad()
        # 重构 phi，仅对 unfrozen_phi 更新
        full_phi = phi.clone()
        full_phi[freeze_mask] = unfrozen_phi  # 更新未冻结的部分
        
        loss = cost_SE_gpu_x(full_phi)
        loss.backward()
        return loss

    optimizer.step(closure)
    
    # 更新冻结掩码
    delta_phi = (phi - phi_last).abs()
    freeze_mask = delta_phi >= freeze_threshold  # 动态更新掩码
    phi_last = phi.clone().detach()
    if step % 2 == 0:
        loss_value = closure().item()
        print(f"Step {i}, Loss: {loss_value:.4f}, Phi: {phi[:5]}")  # 打印前五个参数值
# %%
# 初始设置
# 初始相位猜测和设置
init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)
phi = init_phi
phi.requires_grad = True
freeze_mask = torch.ones_like(phi, dtype=torch.bool).to('cuda')  # 初始时所有元素参与优化
freeze_threshold = 0.0001
phi_last = phi.clone().detach()

# 定义优化器
unfrozen_phi = phi[freeze_mask].clone().detach().requires_grad_(True)
optimizer = torch.optim.LBFGS([unfrozen_phi], lr=0.1)
#%%
for step in range(nb_iter):
    # 检查未冻结部分是否为空
    if freeze_mask.any():
        def closure():
            optimizer.zero_grad()
            # 更新 phi，仅对 unfrozen_phi 应用梯度
            full_phi = phi.clone()
            full_phi[freeze_mask] = unfrozen_phi
            loss = cost_SE_gpu_x(full_phi)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)

        # 使用优化后的 unfrozen_phi 更新 phi
        # 这次不再直接在原地操作，而是使用新的张量来更新 phi
        phi = phi.clone()  # 创建一个新的张量
        phi[freeze_mask] = unfrozen_phi.detach()  # 更新未被冻结的部分

        # 更新冻结掩码
        delta_phi = (phi - phi_last).abs()
        freeze_mask = delta_phi >= freeze_threshold  # 更新冻结掩码
        unfrozen_phi = phi[freeze_mask].clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([unfrozen_phi], lr=0.1)
        phi_last = phi.clone().detach()  # 更新 phi_last
    else:
        print("All parameters are frozen, stopping optimization.")
        break

    # 打印信息
    if step % 1 == 0:
        with torch.no_grad():
            loss_value = cost_SE_gpu_x(phi).item()
        print(f"Step {step}, Loss: {loss_value:.4f}, Phi: {phi[:5]}")  # 打印前五个参数值

# 最终结果
print(f"Optimized Phi: {phi}")

# %%
phi = init_phi
phi.requires_grad=True

# 设置阈值，变化小于这个值的元素将不再参与优化
threshold = 1e-5

# 记录前一轮的 phi
previous_phi = phi.clone()

# 定义优化器
optimizer = torch.optim.LBFGS([phi], lr=0.1)

# 关闭较小变化的元素的梯度
def freeze_small_changes():
    global previous_phi
    with torch.no_grad():
        # 计算 phi 中的变化量
        change = torch.abs(phi - previous_phi)
        
        # 找到变化量小于阈值的元素
        mask = change < threshold
        
        # 只冻结这些变化小的元素的梯度
        phi[mask].requires_grad = False  # 将变化小的元素的 requires_grad 设置为 False

    previous_phi = phi.clone()  # 更新 previous_phi 为当前 phi

def closure():
    optimizer.zero_grad()  # 清空梯度
    loss = cost_SE_gpu_x(phi)  # 计算损失，假设目标函数需要乘以 50
    loss.backward()  # 计算梯度
    return loss

# 执行优化
nb_iter = 50  # 设定最大迭代次数
for step in range(nb_iter):
    previous_phi = phi.clone()  # 在每次迭代开始时更新 previous_phi
    freeze_small_changes()  # 冻结不需要优化的部分
    optimizer.step(closure)  # 使用优化器执行一步更新
    print(cost_SE_gpu_x(phi).item())
# %%

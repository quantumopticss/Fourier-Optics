# -*- coding: utf-8 -*-
#%%
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
def gaussian_line_peak(dim, r0, d, sigma,range,  A=1.0 ,bump_amp=0.02, bump_width_ratio=0.1, save_param=False, device='cuda'):
    """生成带平顶区域局部凸起的高斯线形相位
    Args:
        dim (tuple): 相位图尺寸 (cols, rows)
        r0 (tuple): 中心坐标 (x0, y0)
        d (float): 平顶区域宽度
        sigma (float): 高斯衰减系数
        A (float): 整体幅值
        bump_amp (float): 凸起高度（相对A的比例）
        bump_width_ratio (float): 凸起宽度与d的比例
        save_param (bool): 是否返回参数信息
        device (str): 计算设备
    Returns:
        z (Tensor): 生成的相位图
        param_used (str, optional): 参数信息
    """
    cols, rows = dim
    x = torch.arange(rows, dtype=torch.float32, device=device)
    y = torch.arange(cols, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    x_centered = X - r0[0]
    y_centered = Y - r0[1]
    
    fx = 0.5 * (
        torch.abs(x_centered - d / 2) + 
        torch.abs(x_centered + d / 2) - 
        d
    )

    bump_width = d * bump_width_ratio
    fx_bump =( ((x_centered-0.5*(range[0]+range[1])).pow(2)) / ((bump_width ** 2)))
    bump_mask = (range[0]<=x_centered) & (x_centered<= range[1]) 
    
    # 高斯分布合成
    z = (A * torch.exp( 
        -(fx.pow(2)) / 
        (sigma ** 2))+bump_mask*bump_amp*A*torch.exp( 
        -(fx_bump)))* torch.exp(-(y_centered.pow(2)) / 
        (sigma ** 2))

    if save_param:
        param_str = f"GausLinePeak | dim={dim} | r0={r0} | d={d} | σ={sigma:.2f} | A={A} | bump_amp={bump_amp} | bump_w={bump_width_ratio}"
        return z, param_str
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


# def phase_guess(dim, D, asp, R, ang, B, save_param=False):
#     """
#     Create n x n guess phase: 
#     'D' required radius of shift from origin
#     'asp' aspect ratio of "spreading" for quadratic profile
#     'R' required curvature of quadratic profile
#     'ang' required angle of shift from origin
#     'B' radius of ring in output plane
#     """
#     cols, rows = dim
    
#     # Initialization
#     x = torch.arange(rows) - rows / 2  # Columns
#     y = torch.arange(cols) - cols / 2  # Rows
#     X, Y = torch.meshgrid(x, y, indexing='xy')  # Use meshgrid for 2D arrays
#     z = torch.zeros(size=(rows, cols))

#     # target definition
#     KL = D*((X/shr)*torch.cos(ang)+(Y/shr)*torch.sin(ang));
#     KQ = 3*R*((asp*(torch.pow((X/shr),2))+(1-asp)*(torch.pow((Y/shr),2))));
#     KC = B*torch.pow((torch.pow((X/shr),2)+torch.pow((Y/shr),2)),0.5);
#     z = KC+KQ+KL;
#     z = torch.reshape(z, (rows * cols,))
    
#     if save_param :
#         param_used = "phase_guess | n={0} | D={1} | asp={2} | R={3} | ang={4} | B={5}".format(rows, D, asp, R, ang, B)
#         return z, param_used
#     else :
#         return z

def phase_guess(x_move, y_move, asp, R, ang, B, save_param=False):
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
NT =  [2514,2514]
NT =  [3770,3770]
# NT =  [7540,7540]  # 模型输出平面为 NTxNT 像素阵列 - 更高分辨率
# NT=[2048,2048]
#   ================================================================================================
#   |          激光束
#   ================================================================================================
spix = 0.0125  # SLM 像素大小，单位为 mm
lambda_ = 813e-9
magnification=1
focallength=0.2
focal_spy=lambda_*focallength/(NT[0]*(spix/1000)*magnification)
focal_spx=lambda_*focallength/(NT[1]*(spix/1000)*magnification)
sx = 3.35  # x 轴强度束大小，单位为 mm (1/e^2)
sy = 3.35 # y 轴强度束大小，单位为 mm (1/e^2)
#%%
#   ===   激光幅度   ===============================================
L, Lp = slm.laser_gaussian(dim=N, r0=(0, 0), sigmax=sx/spix, sigmay=sy/spix, A=1.0, save_param=True)
# L=torch.from_numpy(np.sqrt(np.load('inten.npy'))).to('cuda')
# 用于将激光强度的总和匹配到目标强度的总和的归一化
# ===  激光归一化 | 请勿删除  ================
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
L = L * torch.pow(10000 / I_L_tot, 0.5)                      #
I_L_tot = torch.sum(torch.pow(L, 2.))                           #
# ===  激光归一化 | 请勿删除  ================
dx=-40
dy=-40
#   ================================================================================================
#   |          目标幅度、目标相位、加权 cg、加权 i
#   ================================================================================================
# param = [1., round(150e-6/focal_spx), round(25e-6/focal_spx), NT/2., NT/2., 9]  # [d2, sigma, l, roi, roj, C1]
param = [torch.tensor(1.), torch.tensor(round(150e-6/focal_spx)), torch.tensor(round(50e-6/focal_spy)), torch.tensor(dx+NT[0]/2.), torch.tensor(dy+NT[1]/2.), torch.tensor(9)]  # [d2, sigma, l, roi, roj, C1]

d2 = param[0]  # 加权区域宽度
sigma = param[1]  # Laguerre 高斯宽度
l = param[2]
r0i = param[3]
r0j = param[4]
r0 = torch.tensor([r0i, r0j], dtype=torch.float64)   # 模式位置
C1 = param[5]  # 陡度因子

#   ===   目标幅度   ==============================================
# Ta, Tap = slm.target_lg(n=NT, r0=r0, w=sigma, l=l, A=1.0, save_param=True)
Ta, Tap = slm.gaussian_line(dim=NT, r0=r0, d=sigma,sigma=l, A=1.0, save_param=True)
# Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=round(100e-6/focal_spx),sigma=round(80e-6/focal_spx),A=1.0, save_param=True)

# Ta, Tap = gaussian_linex(dim=NT, r0=r0, d2=2,d=sigma,sigma=l, A=1.0, save_param=True)
# Ta, Tap = gaussian_line_peak(dim=NT, r0=r0, d=sigma,sigma=l, A=1.0, range=[sigma/4,sigma/2],save_param=True)
# Ta, Tap =slm.gaussian_top_round(dim=NT, r0=r0,d=18, sigma=4,A=1.0, save_param=True)
#   ===   目标相位   ==================================================
# P, Pp = slm.phase_spinning_continuous(n=NT, r0=r0, save_param=True)
# P, Pp = slm.phase_inverse_square(n=NT[0], r0=r0, save_param=True)
# P, Pp =slm.gaussian_line_phase(n=NT, r0=r0, d=l,sigma=sigma, save_param=True)
P, Pp = slm.phase_flat(dim=NT,v=0, save_param=True)
#   ===   加权 cg   ==================================================
# Weighting, Weightingp = slm.gaussian_top_round(dim=NT, r0=r0, d=2+d2 + sigma, sigma=2, A=1.0, save_param=True)
# Weighting, Weightingp =slm.gaussian_line(dim=NT, r0=r0, d=1.15*sigma,sigma=1.15*l, A=1.0, save_param=True)
Weighting, Weightingp = slm.gaussian_top_round(dim=NT, r0=r0, d=1.5*sigma, sigma=1, A=1.0, save_param=True)#1.35-1.5

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
init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位

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
def calculate_cost(I_Ta, img_nomlz, alpha=0.5, beta=0.1):
    # 计算相对误差
    relative_error = (I_Ta - img_nomlz)/I_Ta
    # 添加平滑项，避免过度修正
    smoothed_cost = torch.exp(alpha * relative_error) - 1
    # 限制修正幅度
    clipped_cost = torch.clamp(smoothed_cost, -beta, beta)
    return 1 + clipped_cost

def cost_SE(phi):
    slm_opt.phi.data = torch.tensor(phi,dtype=torch.float64)# 更新参数
    slm_opt.update_output(NT)
    E_out_a=slm_opt.E_out_amp
    E_out_I = slm_opt.E_out_2
    E_out_ph=slm_opt.E_out_p
    overlap = torch.sum(Ta * E_out_a * Wcg * torch.cos(E_out_ph - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_a * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)
    # cost=torch.sum(torch.pow((Ta_tensor**2 - E_out_I), 2))
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
    overlap = torch.sum(Ta * E_out_amp * Wcg * torch.cos(E_out_p - P))
    overlap = overlap / (torch.pow(torch.sum(torch.pow(Ta, 2)), 0.5) * torch.pow(torch.sum(torch.pow(E_out_amp * Wcg, 2)), 0.5))
    cost = torch.pow(10, C1) * torch.pow((1 - overlap), 2)

    return cost
#   ================================================================================================
#   |          绘图
#   ================================================================================================
p1 = [torch.pow(torch.abs(Ta), 2).cpu(), P.cpu(), Wcg.cpu(), phase_guess_out.cpu(), I_guess_out_amp.cpu()]  # 数据
sz1 = [[], [], [], [], []]  # 缩放轴
t1 = ['ta', 'tp', 'wcg', 'gp', 'ga']  # 标题
v1 = [[], [], [], [], []]  # 限制
c1 = [[], [], [], [], []]  # 颜色
slm.n_plot(p=p1, t=t1, v=v1, c=c1)
plt.show() # 当图形打开时，必须关闭以继续程序
W20 = slm.weighting_value(M=I_Ta, p=0.8, v=0, save_param=False) # 前10%的加权
#   ================================================================================================
#   |          共轭梯度
#   ================================================================================================
nb_iter = 2000
#%%
# mu = np.arctan(200/200)
# D = -2*np.pi*12.5/0.813/200e3*(40*3.45)/np.cos(mu) *4
init_phi, ipp = phase_guess(N, dx,dy, 0.9, 3.7/1000,  0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
# init_phi, ipp = phase_guess(N, -0.18*torch.pi, 0.9, 3.7/1000, torch.tensor(torch.pi/4.4), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位

plt.imshow(I_Ta.detach().cpu()[1750:2000,1750:2000])
plt.show()
slm_opt = slm.SLM(NT=NT,N=N,numb=numb, initial_phi=init_phi, profile_s=L)

plt.imshow(slm_opt.E_out_2.detach().cpu()[1750:2000,1750:2000])
plt.show()

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
            goal=0.001)

#%%
def calculate_cost(I_Ta, img_nomlz, alpha=0.5, beta=0.1):
    # 计算相对误差
    relative_error = (I_Ta - img_nomlz)/I_Ta
    # 添加平滑项，避免过度修正
    smoothed_cost = torch.exp(alpha * relative_error) - 1
    # 限制修正幅度
    clipped_cost = torch.clamp(smoothed_cost, -beta, beta)
    return 1 + clipped_cost
def calculate_cost(I_Ta, img_nomlz, alpha=0.5, beta=0.1):
    # 计算相对误差
    relative_error = (I_Ta - img_nomlz)/I_Ta
    # 添加平滑项，避免过度修正
    smoothed_cost = torch.exp(alpha * relative_error) - 1
    # 限制修正幅度
    clipped_cost = torch.clamp(smoothed_cost, -beta, beta)
    return 1 + clipped_cost
def create_weight_map(I_Ta, edge_weight=0.5):
    # 在边缘区域使用较小的权重
    weight_map = torch.ones_like(I_Ta)
    edge_mask = (I_Ta < 0.1 * I_Ta.max())
    weight_map[edge_mask] = edge_weight
    return weight_map
#%%
def calculate_centroid(image,threshold=80):
    height, width = image.shape
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    
    total_intensity = np.sum(image)
    # print("总光强：",total_intensity)        
    if total_intensity < threshold:
        raise ValueError("图像中未检测到光斑")
    cx = np.sum(xx * image) / total_intensity
    cy = np.sum(yy * image) / total_intensity
    return (cx, cy)

def d4sigma_centroid(image,threshold=0.05):
    h, w = image.shape
    x, y = np.arange(w), np.arange(h)

    image = image.astype(np.float64)
    image -= np.min(image)
    image[image < threshold * np.max(image)] = 0
    total_intensity = np.sum(image)

    (x0,y0)=calculate_centroid(image)
    sigma_x = np.sqrt(np.sum((x - x0)**2 * np.sum(image, axis=0)) / total_intensity)
    sigma_y = np.sqrt(np.sum((y - y0)**2 * np.sum(image, axis=1)) / total_intensity)
    x_min = max(0, int(x0 - 2 * sigma_x))
    x_max = min(w, int(x0 + 2 * sigma_x) + 1)
    y_min = max(0, int(y0 - 2 * sigma_y))
    y_max = min(h, int(y0 + 2 * sigma_y) + 1)
    cropped = image[y_min:y_max, x_min:x_max]

    (x0,y0)=calculate_centroid(cropped)
    x_center = x_min + x0
    y_center = y_min + y0
    return (x_center,y_center,)  
from torch.nn.functional import conv2d


def spatial_smoothing(cost_map, kernel_size=3):
    # 创建double类型的kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float64) / (kernel_size * kernel_size)
    kernel = kernel.to(cost_map.device)
    smoothed_cost = conv2d(cost_map.unsqueeze(0).unsqueeze(0), 
                          kernel, padding=kernel_size//2)
    return smoothed_cost.squeeze()
#%% camera connection, set and initialize
from scipy import ndimage
from sklearn.linear_model import RANSACRegressor

def find_center_by_edge(image, threshold=0.1):
    """通过边缘检测确定平顶光场的中心"""
    # 归一化图像
    norm_image = (image - image.min()) / (image.max() - image.min())
    
    # 使用Sobel算子检测边缘
    dx = ndimage.sobel(norm_image, axis=0)
    dy = ndimage.sobel(norm_image, axis=1)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # 二值化边缘图
    edge_mask = magnitude > threshold * magnitude.max()
    
    # 找到边缘点
    y_coords, x_coords = np.where(edge_mask)
    
    # 使用RANSAC拟合矩形
    points = np.column_stack((x_coords, y_coords))
    
    # 分别拟合上下左右边界
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    return center_x, center_y
def SLM_screen_Correct(SLM_screen):
    LUT_value_for_2pi=215 # 对813nm的LUT数值
    correction=np.array(Image.open('CAL_LSH0802996_810nm.bmp'))
    SLM_screen_Corrected=np.mod(SLM_screen+correction,256)
    SLM_screen_Corrected_LUT=np.around(SLM_screen_Corrected*LUT_value_for_2pi/255).astype('uint8')
    return SLM_screen_Corrected_LUT
def avg_img(rep=20):
    for i in range(rep):
            listx=[]
            image = IDS_Camera.GetImage()
            image=image[100:150-5,120:185]
            center=np.round(d4sigma_centroid(image))
            # center=np.round(find_center_by_edge(image))
            cutted_image=image[int(center[1])-9:int(center[1])+10,int(center[0])-26:int(center[0])+27]
            listx.append(cutted_image.copy())
            stacked_imgs = np.stack(listx, axis=0)  # 形状 (100, h, w)
            mean_img = np.mean(stacked_imgs, axis=0)
    return np.array(mean_img)
#%%

#%%
# ==============================
# 2. 优化：中位数滤波（中间50%数据的中位数）
# ==============================
def truncated_median(data, lower=0.25, upper=0.75):
    """截取中间50%数据后计算中位数"""
    sorted_data = np.sort(data, axis=0)
    n = sorted_data.shape[0]
    start = int(n * lower)
    end = int(n * upper)
    return np.median(sorted_data[start:end], axis=0)

# 逐像素处理（内存优化版本）
# median_img = np.zeros_like(image[int(center[1])-9:int(center[1])+10,int(center[0])-26:int(center[0])+27])
# h,w=median_img.shape
# for i in range(h):
#     for j in range(w):
#         pixel_values = stacked_imgs[:, i, j]
#         median_img[i, j] = truncated_median(pixel_values)

# # ==============================
# # 3. 结果可视化与质量评估
# # 
#%%:
slm_play= Client('192.168.31.144','14000','813 slm play')
IDS_Camera = Client('192.168.31.144','13000','IDS_Camera')

#%%
path='813_3.45c.npy'
tophatscreen=(255*np.load(path)/(2*np.pi))
tophat_screen_Corrected=SLM_screen_Correct(tophatscreen)
# tophatscreen=(255*phase/(2*np.pi))
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
# tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
slm_play.update(tophat_screen_Corrected)
#%%
IDS_Camera.SetBitDepth(12)
# IDS_Camera.SetROI()
x_start = 1200-100
x_end = 1300+100
y_start = 1100-50
y_end = 1250+50

width = x_end - x_start
height = y_end - y_start

success = IDS_Camera.SetROI(x=x_start, y=y_start, width=width, height=height)

#%%
IDS_Camera.SetExposureTime(280)

# %%
IDS_Camera.PrepareAcquisition()
IDS_Camera.AllocAndAnnounceBuffers()
IDS_Camera.StartAcquisition()
#%%
iter=5
for i in range(iter):
    print(i)
    # image = IDS_Camera.GetImage()
    # image=image[100:150-5,120:185]
    # # image=image
    # plt.imshow(image)
    # plt.colorbar()
    # plt.show()
    # print('max:',image.max())
    # print('sum:',image.sum())
    # np.save(f"img{i}.npy",image.copy())
    # # IDS_Camera.StopAcquisition()
    # # IDS_Camera.Close()
    # center=np.round(d4sigma_centroid(image))
    # cutted_image=image[int(center[1])-9:int(center[1])+10,int(center[0])-26:int(center[0])+27]
    # plt.imshow(cutted_image)
    # plt.colorbar()

    cutted_image=avg_img(40)
    plt.imshow(cutted_image)
    plt.colorbar()

    # cutted_image = np.flip(cutted_image,0)
    # cutted_image = image[1002:1022,1100:1192]

    # rotated_image = rotate(cutted_image,angle=-4.8,reshape=False)
    # rotated_image = rotate(cutted_image,angle=-0.2,reshape=False)
    y_numb=cutted_image.shape[0]-1
    x_numb=cutted_image.shape[1]-1
    path1='TA0.npy'
    path2=f'TA{i}.npy'
    Ta=torch.from_numpy(np.load(path1))
    last_Ta=torch.from_numpy(np.load(path2))

    last_Ta_cut=last_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]
    last_I_Ta=torch.pow(torch.abs(last_Ta_cut), 2.)

    Ta_cut=Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]
    I_Ta=torch.pow(torch.abs(Ta_cut), 2.)
    img_nomlz = cutted_image * (I_Ta.sum().item() / cutted_image.sum())
    plt.figure(figsize=(20,20))
    plt.imshow(img_nomlz)
    plt.colorbar()
    plt.show()
    cost=I_Ta/img_nomlz
    # costx = calculate_cost(I_Ta, img_nomlz)
    # cost = spatial_smoothing(costx)
    # cost=I_Ta/img_nomlz*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
    I_new=last_I_Ta*torch.pow(cost,0.5)
    I_new=last_I_Ta*cost
    I_new_nomlz=I_new * (I_Ta.sum().item() / I_new.sum())
    new_Tax=torch.pow(I_new_nomlz,0.5)
    new_Ta=last_Ta
    new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]=new_Tax
    print((new_Ta**2).sum().item())


    # cost=(I_Ta-img_nomlz)
    # # cost=(I_Ta-img_nomlz)*W20.cpu()[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]
    # new_goal=last_I_Ta+0.15*cost
    # new_goal = torch.clamp(new_goal, min=0)
    # new_goal=new_goal*I_Ta.sum().item()/new_goal.sum().item()
    # new_Tax=torch.pow(new_goal,0.5)
    # new_Ta=last_Ta
    # new_Ta[int(r0[0])-int((y_numb)/2):int(r0[0])+int((y_numb)/2)+1,int(r0[1])-int((x_numb)/2):int(r0[1])+int((x_numb)/2)+1]=new_Tax
    # # new_Ta=new_Tay*W20+(1-W20)*torch.from_numpy(np.load(path))
    # # new_Ta=new_Ta*torch.pow(I_L_tot.cpu() / (new_Ta**2).sum(), 0.5)
    # print((new_Ta**2).sum().item())


    # #%%
    # ta_nomlz=np.pow(img_nomlz,0.5)
    # cost=Ta_cut-ta_nomlz
    # new_goal=Ta_cut+1*cost
    # new_goal = torch.clamp(new_goal, min=0)
    # new_Tax=new_goal*torch.pow(I_Ta.sum()/(torch.pow(torch.abs(new_goal), 2.).sum()),0.5)
    # new_Ta=Ta
    # new_Ta[int(r0[0])-int((y_numb-1)/2):int(r0[0])+int((y_numb+1)/2),int(r0[1])-int((x_numb-1)/2):int(r0[1])+int((x_numb+1)/2)]=new_Tax
    np.save(f'TA{i+1}.npy',new_Ta.cpu())
    new_Ta=torch.from_numpy(np.load(f'TA{i+1}.npy'))
    # cols, rows = N
    # new_init_phi=torch.reshape(torch.from_numpy(np.load('813-a-1.npy')),(rows * cols,))
    init_phi, ipp = phase_guess(N, D, 0.9, 3.7/1000, torch.tensor(mu), 0, save_param=True)# init_phi, ipp = slm.phase_guess(N, 0, 0.5, curv/1000, 0, 0, save_param=True)  # 猜测相位
    fft = FourierOp()
    new_Ta=new_Ta.to('cuda')


    def cost_SE_gpu_x(phi):
        zero_frame = torch.zeros((NT[0], NT[1]), dtype=torch.float64)
        zero_matrix = torch.as_tensor(zero_frame, dtype=torch.float64)

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
                nb_iter=100,
                numb=numb,
                slm_opt=slm_opt,
                cost_SE=cost_SE_gpu_x,
                show=True,
                goal=0.01)

    np.save(f'813ad{i+1}.npy',cg1.slm_phase_end)
    path=f'813ad{i+1}.npy'
    tophatscreen=(255*np.load(path)/(2*np.pi))
    tophat_screen_Corrected=SLM_screen_Correct(tophatscreen)
    # tophatscreen=(255*phase/(2*np.pi))
    # tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
    # tophat_screen_Corrected=(215*tophatscreen/255).astype('uint8')
    slm_play.update(tophat_screen_Corrected)

#%%

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
path='813_test.npy'
tophatscreen=255*np.load(path)/(2*np.pi)
# tophatscreen=255*np.array(cg1.slm_phase_end)/(2*np.pi)
tophat_screen_Corrected=IMGpy.SLM_screen_Correct(tophatscreen)
slmx = slmpy.SLMdisplay(monitor=1,isImageLock = True)
slmx.updateArray(tophat_screen_Corrected)
# %%
slmx.close()

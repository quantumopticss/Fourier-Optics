import torch as th
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = "cuda"

## 1d demo
class flat_top_tail(nn.Module):
    def __init__(self,n_order,m_order):
        super().__init__()
        self.n_order = n_order
        self.xlist = th.linspace(-2,2,2*n_order+1,requires_grad=False,device=device)
        self.ylist = th.linspace(-2,2,2*n_order+1,requires_grad=False,device=device)
        self.xx, self.yy = th.meshgrid(self.xlist,self.ylist,indexing="ij")
        
        self.target = th.zeros_like(self.xx,requires_grad=False,device=device)
        self.target[n_order-m_order:n_order+m_order+1,n_order-m_order:n_order+m_order+1] = 1.0
        # self.target[(self.xx**2 + self.yy**2) <= 0.8**2] = 1.0
        self.side_target = 1.0 - self.target
        # self.side_target = th.tanh(100*(th.abs(self.xlist) - 2*m_order/n_order))
        # g0 = th.ones_like(self.xlist)/(th.abs(self.xlist)+3.0)
        self.sides = nn.Parameter(th.exp(-3*(self.xx**2+self.yy**2)))
        
    def power_cost(self,sides,s_target,set=0.075):
        eff = th.sum(sides**2,dim = None)/th.sum(s_target**2,dim = None)
        if eff <= set:
            return 0.0
        else:
            return 60*th.exp(5*(eff-set)**2)

    def forward(self,limit):
        sides = th.abs(self.sides*self.side_target)
        # sides = sides/sides.max()
        
        s_target = self.target + sides
        p_f_target = th.abs(th.fft.fftshift(th.fft.fft2(s_target),dim = (0,1)))**2
        
        total_power = th.sum(p_f_target,dim = None)
        power = th.sum(p_f_target[self.n_order-limit:self.n_order+limit+1,self.n_order-limit:self.n_order+limit+1],dim = None)
        cost_momentum = - 200 * (power/total_power)
        cost_power = self.power_cost(sides, self.target)
        
        return sides,s_target, cost_momentum + cost_power

N = 800
M = 100
model = flat_top_tail(n_order = N,m_order = M)
model.to(device)
opt = th.optim.AdamW(model.parameters(),lr = 5e-3)

model.train()
cost_list = []
limit = 2
for i in range(4000):
    
    sides,s_target,cost = model(limit)
    
    opt.zero_grad()
    cost.backward()
    opt.step()
    
    if (i+1)%50 == 0:
        print(i,cost)
        cost_list.append(cost.cpu().detach().numpy())

th.save(model.state_dict(),"test.pth")

# plt.plot(cost_list)

model.cpu()

target_raw = model.target.cpu().detach().numpy()
f_numpy = s_target.cpu().detach().numpy()
plt.subplot(3,2,2)
plt.imshow(f_numpy)
plt.title('new')

plt.subplot(3,2,3)
plt.imshow(target_raw)
plt.title("raw")

plt.subplot(3,2,4)
plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(f_numpy),axes = (0,1))))[N-10*limit:N+10*limit+1,N-10*limit:N+10*limit+1],label = "f_raw")
plt.title("f_new")

plt.subplot(3,2,1)
plt.plot((np.abs(np.fft.fftshift(np.fft.fft2(f_numpy),axes = (0,1))))[N,N-10*limit:N+10*limit+1],label = "new")
plt.plot((np.abs(np.fft.fftshift(np.fft.fft2(target_raw),axes = (0,1))))[N,N-10*limit:N+10*limit+1],label = "raw")
plt.legend()

plt.subplot(3,2,5)
plt.imshow((np.abs(np.fft.fftshift(np.fft.fft2(target_raw),axes = (0,1))))[N-10*limit:N+10*limit+1,N-10*limit:N+10*limit+1],label = "f_new")
plt.title("f_raw")

plt.show()

xx = model.xx.cpu().detach().numpy()
yy = model.yy.cpu().detach().numpy()
np.save("xx.txt",xx)
np.save("yy.txt",yy)
np.save("data.txt",f_numpy)
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
        
        self.target = th.zeros_like(self.xlist,requires_grad=False,device=device)
        self.target[n_order-m_order:n_order+m_order+1] = 1.0
        self.side_target = 1.0 - self.target
        # self.side_target = th.tanh(100*(th.abs(self.xlist) - 2*m_order/n_order))
        # g0 = th.ones_like(self.xlist)/(th.abs(self.xlist)+3.0)
        self.sides = nn.Parameter(th.exp(-2*self.xlist**2))
        
    def power_cost(self,sides,s_target):
        eff = th.sum(sides**2)/th.sum(s_target**2)
        if eff <= 0.05:
            return 0.0
        else:
            return 60*th.exp(10*(eff-0.03)**2)

    def forward(self,limit):
        sides = th.abs(self.sides*self.side_target)
        # sides = sides/sides.max()
        
        s_target = self.target + sides
        p_f_target = th.abs(th.fft.fftshift(th.fft.fft(s_target)))**2
        
        total_power = th.sum(p_f_target)
        power = th.sum(p_f_target[self.n_order-limit:self.n_order+limit+1])
        cost_momentum = - 200 * (power/total_power)
        cost_power = self.power_cost(sides, self.target)
        
        return sides,s_target, cost_momentum + cost_power

N = 500
M = 50
model = flat_top_tail(n_order = N,m_order = M)
model.to(device)
opt = th.optim.AdamW(model.parameters(),lr = 6e-3)

model.train()
cost_list = []
limit = 4
for i in range(3000):
    
    sides,s_target,cost = model(limit)
    
    opt.zero_grad()
    cost.backward()
    opt.step()
    
    if (i+1)%100 == 0:
        print(i,cost)
        cost_list.append(cost.cpu().detach().numpy())

th.save(model.state_dict(),"test.pth")
plt.subplot(2,2,1)
plt.plot(cost_list)

target_raw = model.target.cpu().detach().numpy()
f_numpy = s_target.cpu().detach().numpy()
plt.subplot(2,2,2)
plt.plot(f_numpy,label = "new")
plt.plot(target_raw,label = "raw")
plt.legend()

plt.subplot(2,2,3)
plt.plot((np.abs(np.fft.fftshift(np.fft.fft(f_numpy))))[N-10*limit:N+10*limit+1],label = "f_raw")
plt.plot((np.abs(np.fft.fftshift(np.fft.fft(target_raw))))[N-10*limit:N+10*limit+1],label = "f_new")
plt.legend()

plt.subplot(2,2,4)
plt.plot(sides.cpu().detach().numpy())

plt.show()
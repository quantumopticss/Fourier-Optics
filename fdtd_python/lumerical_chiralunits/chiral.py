import sys
import time
import numpy as np
import os
from matplotlib import pyplot as plt
sys.path.append(r"D:\Program Files\Lumerical\v241\api\python")     # Default windows lumapi path
import lum_fdtd

## test
if __name__ == '__main__':
    
    unit = 1e-6
    W = 1.2*unit
    Num = 10
    T = 0.5

    M_matrix = np.zeros([Num,Num],dtype = int)
    M_matrix[2:7+1:1,2:5+1:1] = 1

    ylist = ( np.arange(0,Num,1) - (Num-1)/2 )* (W/Num)
    xlist = -ylist

    location_X, location_Y = np.meshgrid(ylist,xlist)

    period = 2*unit
    h = 1*unit
    epoch = 2
    score_old = 0

    for n in range(epoch):

        # data record
        print(f'process: {n+1}/{epoch},' + f'score = {score_old}')
        # ################## 参数定义 ##############
        # 基础结构
        unity_sim = lum_fdtd.LumSimulation(hide=False)
        unit = unity_sim.set_unit(unit=unit)

        substrate_thickness = 1 * unit  # SiO2层衬底厚度
        source_z = -0.3*unit        # location of the source
        Tz2 = 1.6 * unit              # length of the light propagation
        Tz1 = -0.2*substrate_thickness    # location of the incident monitor
        ####### wavelength setting
        wl_max = 0.538 * unit               # maximum wavelength 
        wl_min = 0.538 * unit               # minimum wavelength
        nfreq = 1                        # sampling points of the wavelength area
        save_name = 'chiral units'        # 仿真文件名，不用加后缀.fsp

        unity_sim.set_sim_paras(wl_max=wl_max, wl_min=wl_min, T_z=Tz2, nfreq=nfreq, resolution=40) # ?????

        # # 材料定义
        unity_sim.define_material('SiO2', 1.45)   # 定义材料
        unity_sim.define_material('TiO2', 2.66)     # 定义材料
        unity_sim.define_material('air', 1, color=[0.501961, 0.501961, 1], opacity=0.65)
        M = ['air','TiO2']
        Color = ['blue','red']

        # 添加 SiO2 衬底
        unity_sim.add_block(size=(1.2*period, 1.2*period, substrate_thickness), center=(0, 0, -substrate_thickness/2),
                            material_name='SiO2', name='substrate', color='Grey')

        # 创建超表面单元
        index = np.random.randint(0,Num,size=[2])
        M_matrix[index[0],index[1]] = 1 - M_matrix[index[0],index[1]]

        i = 0
        while(i<Num):
            j = 0
            while(j<Num):
                k = M_matrix[i,j]
                if k == 1:
                    unity_sim.add_block(size=(W/Num, W/Num, h), center=(location_X[i,j], location_Y[i,j], h/2), material_name=M[k], name=(f'M_({i},{j})-' + M[k]), color=Color[k])
                j += 1
            i += 1

        # 添加源和监视器
        unity_sim.add_source(size=(period, period, 0), center=(0, 0, source_z),amplitude = np.sqrt(2), jones = [1,0], source_type = "PLaneSource",name = '1')
        unity_sim.add_monitor(size=(period, period, 0), center=(0, 0, Tz2), name='Transmit_monitor')

        # 添加 FDTD
        zmax = Tz2 + 0.2 * unit
        zmin = -0.4 * unit
        z_span = zmax - zmin
        z_cen = (zmin + zmax) / 2
        unity_sim.add_sim_area(size=(period, period, z_span), center=(0, 0, z_cen),
                            boundary='Periodic', mesh_accuracy=3, sim_time=4000e-15)

        # 仿真运行
        '''
        unity_sim.run(save_name=save_name)
        result_t = unity_sim.get_result_monitor_field(name='Transmit_monitor')

        unity_sim.quit()

        # 结果处理 - 透射率        
        ex = result_t["Ex"]
        ey = result_t["Ey"]
        Ex = np.sum(ex,axis = None)/np.size(ex)
        Ey = np.sum(ey,axis = None)/np.size(ey)

        A_lcp = (Ex + 1j*Ey)/np.sqrt(2)
        A_rcp = (Ex - 1j*Ey)/np.sqrt(2)

        I_lcp = np.abs(A_lcp)**2
        I_rcp = np.abs(A_rcp)**2

        score = (I_rcp - I_lcp)/(I_rcp + I_lcp)
        if score > score_old:
            pass
        else:
            if np.random.rand() > np.exp( -(score_old - score)/T ):
                M_matrix[index[0],index[1]] = 1 - M_matrix[index[0],index[1]]
        score_old = score

        #fig, ax = plt.subplots(1, 1)
        #plt.plot(wl, abs(T_LCP), 'b--', label='LCP')

        #plt.xlabel("wavelength (μm)")
        #plt.legend(loc="upper right")
        #plt.savefig(f"chiral_data/LCP/LCP_{i}.png")
        #plt.close()
        
        os.remove(f'{save_name}.fsp')
        os.remove(f'{save_name}_p0.log')

        if (n)%1 == 0:
            np.savetxt(f'material matrix_{n}.txt',M_matrix)
        
        time.sleep(1)
        '''

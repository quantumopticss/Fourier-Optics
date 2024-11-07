import sys
import time
import numpy as np
import os
from matplotlib import pyplot as plt
sys.path.append(r"D:\Program Files\Lumerical\v241\api\python")     # Default windows lumapi path
import lum_fdtd

if __name__ == '__main__':
    unit = 1e-6
    alist = np.linspace(30,150,60)*unit/1e3
    blist = np.linspace(30,150,60)*unit/1e3
    period = 2*unit
    h = 1*unit

    anglelist = np.linspace(0,np.pi/2,1,endpoint = False)
    for theta in anglelist:
        for a in alist:
            for b in blist:
                # data record
                print(theta,a,b)
                # ################## 参数定义 ##############
                # 基础结构
                unity_sim = lum_fdtd.LumSimulation(hide=False)
                unit = unity_sim.set_unit(unit=unit)

                substrate_thickness = 1 * unit  # SiO2层衬底厚度
                source_z = -0.3*unit        # location of the source
                Tz2 = 2 * unit              # location of the transmite monitor
                Tz1 = -0.2*substrate_thickness    # location of the incident monitor

                wl_max = 0.538 * unit               # 设置仿真波长
                wl_min = 0.538 * unit
                nfreq = 1
                save_name = 'cylinder_meta'        # 仿真文件名，不用加后缀.fsp

                unity_sim.set_sim_paras(wl_max=wl_max, wl_min=wl_min, T_z=Tz2, nfreq=nfreq, resolution=50) # ?????

                # # 材料定义
                unity_sim.define_material('SiO2', 1.45)   # 定义材料
                unity_sim.define_material('TiO2', 2.66)     # 定义材料
                unity_sim.define_material('air', 1, color=[0.501961, 0.501961, 1], opacity=0.65)

                # 添加 SiO2 衬底
                unity_sim.add_block(size=(period, period, substrate_thickness), center=(0, 0, -substrate_thickness/2),
                                    material_name='SiO2', name='substrate', color='Grey')

                # 创建超表面单元
                unity_sim.add_block(size=(a, b, h), center=(0, 0, h/2), material_name='TiO2', name='meta1', color='Red')

                # 添加源和监视器
                unity_sim.add_source(size=(period, period, 0), center=(0, 0, source_z),amplitude = 1, jones = [1,0], source_type = "PLaneSource",name = '1')
                
                unity_sim.add_monitor(size=(period, period, 0), center=(0, 0, Tz1), name='Incident_monitor')
                unity_sim.add_monitor(size=(period, period, 0), center=(0, 0, Tz2), name='Transmit_monitor')

                # 添加 FDTD
                zmax = Tz2 + 0.2 * unit
                zmin = -0.6 * unit
                z_span = zmax - zmin
                z_cen = (zmin + zmax) / 2
                unity_sim.add_sim_area(size=((np.max(alist,axis = None)+0.5*a), (np.max(blist,axis = None)+0.5*b), z_span), center=(0, 0, z_cen),
                                    boundary='Periodic', mesh_accuracy=3, sim_time=5000e-15)

                # 仿真运行
                unity_sim.run(save_name=save_name)
                result_i = unity_sim.get_result_monitor_field(name='Incident_monitor')
                result_t = unity_sim.get_result_monitor_field(name='Transmit_monitor')

                unity_sim.quit()

                # 结果处理 - 透射率
                ### wl = result_LCP["wl"]
                
                ex = result_t["Ex"]
                ex = np.round(ex,4)
                ey = result_t["Ey"]
                ey = np.round(ey,4)

                nx = result_i['Ex']
                ny = result_i['Ey']
                Ne = np.sqrt(np.sum(np.abs(nx)**2+np.abs(ny)**2,axis = None)/np.size(nx))

                Exx = (ex*np.cos(theta) - ey*np.sin(theta))/Ne
                Eyy = (ey*np.cos(theta) + ex*np.sin(theta))/Ne

                #fig, ax = plt.subplots(1, 1)
                #plt.plot(wl, abs(T_LCP), 'b--', label='LCP')

                #plt.xlabel("wavelength (μm)")
                #plt.legend(loc="upper right")
                #plt.savefig(f"chiral_data/LCP/LCP_{i}.png")
                #plt.close()

                os.remove(f'{save_name}.fsp')
                os.remove(f'{save_name}_p0.log')

                np.savetxt(("ab/" + "EXresult" + f"_theta = {theta:.2f},a = {(a*1e9):.2f},b = {(b*1e9):.2f}.txt"), Exx)
                np.savetxt(("ab/" + "EYresult" + f"_theta = {theta:.2f},a = {(a*1e9):.2f},b = {(b*1e9):.2f}.txt"), Eyy)
                
                time.sleep(1)

import sys
import time

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
sys.path.append(r"D:\Program Files\Lumerical\v241\api\python")     # Default windows lumapi path
import lumapi

class BaseSimulation(object):
    materials = {}
    models = []
    task_id = ''
    wl_max = 1.6
    wl_min = 1.1
    nfreq = 200
    T_z = 1.9
    resolution = 50
    auto_shutoff_min = 1e-9
    def __init__(self):
        pass

    def set_sim_paras(self, **kwargs):
        if 'wl_max' in kwargs:
            self.wl_max = kwargs.get('wl_max')
        if 'wl_min' in kwargs:
            self.wl_min = kwargs.get('wl_min')
        if 'nfreq' in kwargs:
            self.nfreq = kwargs.get('nfreq')
        if 'T_z' in kwargs:
            self.T_z = kwargs.get('T_z')
        if 'resolution' in kwargs:
            self.resolution = kwargs.get('resolution')
        if 'auto shutoff min' in kwargs:
            self.auto_shutoff_min = kwargs.get('auto shutoff min')

    def get_useful_para_dic(self, paras, **kwargs):
        useful_para_dic = {}
        for para in paras:
            if para in kwargs:
                useful_para_dic[para] = kwargs.get(para)
        return useful_para_dic

    def define_material(self, name, n, **kwargs):
        ''' 定义材质
        name: 材质名称
        n: 折射率
        '''
        pass

    def add_block(self, size, center, material_name, **kwargs):
        '''
        size: 长度为3的数组
        center: 长度为3的数组
        material_name: 使用的材质名称
        '''
        pass

    def add_cylinder(self, center, radius, height, material_name, **kwargs):
        '''
        center: 长度为3的数组
        radius: 半径
        height: 高度
        material_name: 材质名称
        '''
        pass

    def combine_model(self, model1, model2, **kwargs):
        pass

    def add_source(self, center, size, **kwargs):
        pass

    def add_sim_area(self, center, size, **kwargs):
        pass

    def add_middle_monitor(self, center, size, **kwargs):
        pass

    def add_monitor(self, center, size, **kwargs):
        pass

    def run(self, **kwargs):
        pass

    def get_middle_result(self):
        pass

    def get_result(self, **kwargs):
        pass

class LumSimulation(BaseSimulation):
    def __init__(self, hide=False):
        super(BaseSimulation, self).__init__()
        self.fdtd = lumapi.FDTD(hide=hide)

        self.fdtd.switchtolayout()
        self.fdtd.selectall()
        self.fdtd.delete()

        self.unit = 1e-6

    def set_unit(self, **kwargs):
        ''' 设置单位 (为与cst统一而设置)
        parameters: unit: 长度单位, dafault=1e-6
        return:
        '''
        scale = kwargs.get('unit', 1e-6)
        self.unit = scale
        return scale

    def set_sim_paras(self, **kwargs):
        super(LumSimulation, self).set_sim_paras(**kwargs)

    def define_material(self, name, n, **kwargs):
        """定义材料的折射率
            name：类型：str，自定义材料名称
            n:材料的折射率
        """
        self.materials[name] = n

    def nk_material(self, name, data, **kwargs):
        """
        自定义n,k模型，传入材料
        par:
            name:材料名称，str
            data: 从数据库中传入的材料数据, typy: [dict], key: [wave, n, k]
        """
        w = data['wave']
        w = w * 1e-6     # 材料库数据的波长单位是 um
        n = data['n']
        k = data['k']
        data_len = len(w)
        # 考虑没有k数据的情况
        if not all(k):
            k = [0] * data_len

        f = []
        eps = []
        for lda in range(len(w)):
            x = 3 * 1e8 / w[lda]
            f.append(x)
            a = (n[lda] ** 2 - k[lda] ** 2) + n[lda] * k[lda] * 2j
            eps.append(a)

        # create example permittivity vector
        sampledData = np.array(list(zip(f, eps))) # collect f and eps in one matrix
        matName = name
        temp = self.fdtd.addmaterial("Sampled data")
        self.fdtd.setmaterial(temp, "name", matName)  # rename material
        self.fdtd.setmaterial(matName, "max coefficients", 2)  # set the number of coefficients
        self.fdtd.setmaterial(matName, "sampled data", sampledData)  # load the sampled data matrix

        self.materials[name] = name

    def add_block(self, size, center, material_name, **kwargs):
        """ 定义方块模型尺寸
        parameter:
            center:  (x,y,z),中心坐标
            size: (x_span,y_span,z_span),沿x,y,z轴尺寸大小
            material_name: 定义材料,类型：str
        **kwargs：
            name:定义模型名称
            color：定义模型材料颜色
            opacity:定义模型透明度
            line：True(default),是否线框
        """
        if len(size) != 3:
            raise Exception("size的长度应为3")
        if len(center) != 3:
            raise Exception("center的长度应为3")
        if material_name not in self.materials:
            raise Exception("材质未定义")
        self.fdtd.addrect()
        self.fdtd.set('name', kwargs.get('name', "box"))
        self.fdtd.set('x', center[0])
        self.fdtd.set('x span', size[0])
        self.fdtd.set('y', center[1])
        self.fdtd.set('y span', size[1])
        self.fdtd.set('z', center[2])
        self.fdtd.set('z span', size[2])
        if type(self.materials[material_name]) is str:
            self.fdtd.set('material', self.materials[material_name])
        else:
            self.fdtd.set('material', "<Object defined dielectric>")
            self.fdtd.set('index', self.materials[material_name])

        if 'mesh_order' in kwargs:
            self.fdtd.set("override mesh order from material database", 1)
            self.fdtd.set("mesh order", kwargs.get('mesh_order'))


    def add_cylinder(self, center, radius, height, material_name,**kwargs):
        """定义圆柱模型尺寸
        parameter:
            center:  (x,y,z),中心坐标
            radius: 圆柱半径
            height：圆柱高度
            material_name: 定义材料，类型：str
        **kwargs：
            name:定义模型名称
            color：定义模型材料颜色
            opacity:定义模型透明度
            line：True(default),是否线框
        """
        if len(center) != 3:
            raise Exception("center的长度应为3")
        if material_name not in self.materials:
            raise Exception("材质未定义")
        self.fdtd.addcircle()
        self.fdtd.set('name', kwargs.get('name', "cylinder"))
        self.fdtd.set("x", center[0])
        self.fdtd.set("y", center[1])
        self.fdtd.set("radius", radius)
        self.fdtd.set("z", center[2])
        self.fdtd.set("z span", height)
        if type(self.materials[material_name]) is str:
            self.fdtd.set('material', self.materials[material_name])
        else:
            self.fdtd.set('material', "<Object defined dielectric>")
            self.fdtd.set('index', self.materials[material_name])

        if 'mesh_order' in kwargs:
            self.fdtd.set("override mesh order from material database", 1)
            self.fdtd.set("mesh order", kwargs.get('mesh_order'))

    def combine_models(self, model1, model2, **kwargs):
        """组合模型
        parameter:
            model1:
            mode2:
        """
        pass

    def add_source(self, center, size, **kwargs):
        """定义光源
        parameter:
            center:  (x,y,z),中心坐标
            size: (x_span,y_span,z_span),沿x,y,z轴尺寸大小
        **kwargs：
            name:定义模型名称
            color：定义模型材料颜色，Red(default)
            opacity:定义模型透明度
            line：True(default),是否线框
            mode: TE(default),TM, LCP, RCP
            source_type: 源类型   PlaneSource(dafault): 上方入射, 支持TE/TM/LCP/RCP
                          GaussSource: 下方入射, 支持 TE/TM
            w0: 束腰半径
            direction: 光的入射方向，default='backward'
        """
        if len(size) != 3:
            raise Exception("size的长度应为3")
        if len(center) != 3:
            raise Exception("center的长度应为3")

        mode = kwargs.get('mode', 'TE')
        source_type = kwargs.get('source_type', 'PlaneSource')
        direction = kwargs.get('direction', 'backward')

        if source_type == 'GaussSource':
            if 'w0' in kwargs:
                w0 = kwargs.get('w0')
                self.fdtd.addgaussian()
                self.fdtd.set("name", "source_gauss")
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)
                if mode == 'TE':
                    self.fdtd.set("polarization angle", 0)
                elif mode == 'TM':
                    self.fdtd.set("polarization angle", 90)
                else:
                    print("GaussSource only set TE/TM mode, please choose right mode ! ")
                self.fdtd.set("waist radius w0", w0)
            else:
                print('please input w0 (waist radius)')

        elif source_type == 'PlaneSource':
            if mode == 'TE' or mode == 'TM':
                self.fdtd.addplane()
                self.fdtd.set("name", kwargs.get('name', "source"))
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)
                if mode == "TE":
                    self.fdtd.set("polarization angle", 90)
                else:
                    self.fdtd.set("polarization angle", 0)

            elif mode == 'LCP':
                self.fdtd.addplane()
                self.fdtd.set("name", "source_X")
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)
                self.fdtd.set("Phase", 90)

                self.fdtd.addplane()
                self.fdtd.set("name", "source_Y")
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)
                self.fdtd.set("polarization angle", 90)

            elif mode == 'RCP':
                self.fdtd.addplane()
                self.fdtd.set("name", "source_X")
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)

                self.fdtd.addplane()
                self.fdtd.set("name", "source_Y")
                self.fdtd.set("injection axis", "z")
                self.fdtd.set("direction", direction)
                self.fdtd.set("x", center[0])
                self.fdtd.set("x span", size[0])
                self.fdtd.set("y", center[1])
                self.fdtd.set("y span", size[1])
                self.fdtd.set("z", center[2])
                self.fdtd.set("wavelength start", self.wl_min)
                self.fdtd.set("wavelength stop", self.wl_max)
                self.fdtd.set("polarization angle", 90)
                self.fdtd.set("Phase", 90)

            else:
                print('Please choose in TE/TM/LCP/RCP ! ')

        else:
            print("Please choose PlaneSource or GaussSource ! ")

    def add_sim_area(self, center, size, **kwargs):
        """ 定义仿真区域
        parameter:
            center:  (x,y,z),中心坐标
            size: (x_span,y_span,z_span),沿x,y,z轴尺寸大小
        **kwargs：
            name:定义模型名称
            boundary: 边界条件: Periodic(default)/ PML/ Symmetric_TE/ Symmetric_TM
                               或者输入 {x_bc, y_bc, z_bc}指定各自方向的边界条件, 必须输入x_bc, y_bc
            color：定义模型材料颜色，Orange(default)
            opacity:定义模型透明度, 0.4(default)
            line：True(default),是否线框
            sim_time: 仿真时间，10000 * 1e-15 (default)
            mesh_accuracy：网格精度，2 (default)
            mesh_dx:网格x方向的精度，6e-9 (default)
            mesh_dy:网格y方向的精度, 6e-9 (default)
        """
        if len(size) != 3:
            raise Exception("size的长度应为3")
        if len(center) != 3:
            raise Exception("center的长度应为3")

        boundary = kwargs.get('boundary')

        self.fdtd.addfdtd()
        self.fdtd.set('dimension', '3D')
        if 'sim_time' in kwargs:
            self.fdtd.set("simulation time", kwargs.get('sim_time'))
        self.fdtd.set('index', 1)
        self.fdtd.set('mesh accuracy', kwargs.get('mesh_accuracy', 2))
        self.fdtd.set("mesh refinement", 1)
        self.fdtd.set('x', center[0])
        self.fdtd.set('x span', size[0])
        self.fdtd.set('y', center[1])
        self.fdtd.set('y span', size[1])
        self.fdtd.set('z', center[2])
        self.fdtd.set('z span', size[2])
        self.fdtd.set('auto shutoff min', self.auto_shutoff_min)
        self.fdtd.set("allow symmetry on all boundaries", 1)
        if boundary == "Symmetric_TM":
            self.fdtd.set("x min bc", "Anti-Symmetric")
            self.fdtd.set("x max bc", "Anti-Symmetric")
            self.fdtd.set("y min bc", "Symmetric")
            self.fdtd.set("y max bc", "Symmetric")
        elif boundary == "Symmetric_TE":
            self.fdtd.set("y min bc", "Anti-Symmetric")
            self.fdtd.set("y max bc", "Anti-Symmetric")
            self.fdtd.set("x min bc", "Symmetric")
            self.fdtd.set("x max bc", "Symmetric")
        elif boundary == "PML":
            self.fdtd.set("x min bc", "PML")
            self.fdtd.set("y min bc", "PML")
        elif boundary == 'Periodic':
            self.fdtd.set("x min bc", "Periodic")  # 设置边界条件
            self.fdtd.set("y min bc", "Periodic")
        else:
            self.fdtd.set("x min bc", boundary.get('x_bc'))
            self.fdtd.set("y min bc", boundary.get('y_bc'))
            self.fdtd.set("z min bc", boundary.get('z_bc', 'PML'))

        if 'mesh_dx' in kwargs and 'mesh_dy' in kwargs:
            # add mesh
            self.fdtd.addmesh()
            self.fdtd.set("name", "mesh")
            # set dimension
            self.fdtd.set('x', center[0])
            self.fdtd.set('x span', size[0])
            self.fdtd.set('y', center[1])
            self.fdtd.set('y span', size[1])
            self.fdtd.set('z', center[2])
            self.fdtd.set('z span', size[2])
            # enable in X direction and disable in Y,Z directions
            self.fdtd.set("override x mesh", 1)
            self.fdtd.set("override y mesh", 1)
            self.fdtd.set("override z mesh", 0)
            # restrict mesh by defining maximum step size
            self.fdtd.set("set maximum mesh step", 1)
            self.fdtd.set("dx", kwargs.get('mesh_dx'))
            self.fdtd.set("dy", kwargs.get('mesh_dy'))

    def add_monitor(self, center, size, **kwargs):
        """定义功率监视器区域
        parameter:
            center:  (x,y,z),中心坐标
            size: (x_span,y_span,z_span),沿x,y,z轴尺寸大小
        **kwargs：
            name:定义模型名称
            color：定义模型材料颜色，Red(default)
            opacity:定义模型透明度
            line：True(default),是否线框
        return:
            监视器的名称
        """
        if len(size) != 3:
            raise Exception("size的长度应为3")
        if len(center) != 3:
            raise Exception("center的长度应为3")

        self.fdtd.addpower()
        self.fdtd.set('name', kwargs.get("name", "T"))
        if size[2] == 0:
            self.fdtd.set('monitor type', '2D Z-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('x span', size[0])
            self.fdtd.set('y', center[1])
            self.fdtd.set('y span', size[1])
            self.fdtd.set('z', center[2])
        elif size[0] == 0:
            self.fdtd.set('monitor type', '2D X-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('y', center[1])
            self.fdtd.set('y span', size[1])
            self.fdtd.set('z', center[2])
            self.fdtd.set('z span', size[2])
        elif size[1] == 0:
            self.fdtd.set('monitor type', '2D y-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('x span', size[2])
            self.fdtd.set('y', center[1])
            self.fdtd.set('z', center[2])
            self.fdtd.set('z span', size[2])
        self.fdtd.set("override global monitor settings", 1)
        self.fdtd.set("frequency points", self.nfreq)

    def add_monitor_field(self, center, size, **kwargs):
        """定义电场监视器区域
        parameter:
            center:  (x,y,z),中心坐标
            size: (x_span,y_span,z_span),沿x,y,z轴尺寸大小
        **kwargs：
            name:定义模型名称
            color：定义模型材料颜色，Red(default)
            opacity:定义模型透明度
            line：True(default),是否线框
        """
        if len(size) != 3:
            raise Exception("size的长度应为3")
        if len(center) != 3:
            raise Exception("center的长度应为3")

        self.fdtd.addprofile()
        self.fdtd.set('name', kwargs.get("name", "E_field"))
        if size[2] == 0:
            self.fdtd.set('monitor type',  '2D Z-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('x span', size[0])
            self.fdtd.set('y', center[1])
            self.fdtd.set('y span', size[1])
            self.fdtd.set('z', center[2])
        elif size[0] == 0:
            self.fdtd.set('monitor type', '2D X-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('y', center[1])
            self.fdtd.set('y span', size[1])
            self.fdtd.set('z', center[2])
            self.fdtd.set('z span', size[2])
        elif size[1] == 0:
            self.fdtd.set('monitor type', '2D y-normal')
            self.fdtd.set('x', center[0])
            self.fdtd.set('x span', size[2])
            self.fdtd.set('y', center[1])
            self.fdtd.set('z', center[2])
            self.fdtd.set('z span', size[2])

        self.fdtd.set("override global monitor settings", 1)
        self.fdtd.set("frequency points", self.nfreq)

    def run(self,  **kwargs):
        """仿真运行
        **kwargs:
            name:后台保存文件.fsp名称
        """
        name = kwargs.get('save_name', 'meta_model')
        self.fdtd.save("%s" % name)
        self.fdtd.save(f"{name}.fsp")
        self.fdtd.run()

    def get_result_monitor(self, **kwargs):
        """从功率监视器提取数据
        parameter:
            name:  监视器，add_monitor()
            unit: 单位, default=1e-6, 则此时若 wl=1, 则表示 wl = 1e-6m (1um)
        return：
            power:从监视器提取的数据T/R
            f:频率点
        """
        unit = self.unit

        T = self.fdtd.transmission(kwargs.get("name", "T"))
        f = self.fdtd.getdata(kwargs.get("name", "T"), "f")
        wl = 3 * 1e8 / f / unit

        return {'power': T, 'f': f, 'wl': wl}

    def get_result_monitor_field(self, **kwargs):
        """从电场监视器提取数据
        return：
            Ex:电场x的分量
            Ey:电场y的分量
            Ez:电场z的分量
        """
        # unit = self.unit
        Ex = self.fdtd.getdata(kwargs.get("name", "E_field"), "Ex")
        Ey = self.fdtd.getdata(kwargs.get("name", "E_field"), "Ey")
        Ez = self.fdtd.getdata(kwargs.get("name", "E_field"), "Ez")
        f = self.fdtd.getdata(kwargs.get("name", "T"), "f")

        xx = self.fdtd.getdata(kwargs.get("name", "E_field"), "x")
        yy = self.fdtd.getdata(kwargs.get("name", "E_field"), "y")
        zz = self.fdtd.getdata(kwargs.get("name", "E_field"), "z")

        wl = 3 * 1e8 / f

        data_Ex = np.squeeze(np.array(np.real(Ex)))
        data_Ey = np.squeeze(np.array(np.real(Ey)))
        data_Ez = np.squeeze(np.array(np.real(Ez)))

        # b = data_Ex[:, :, 0]
        # c = data_Ex[:, :, 0].T    # 这个才是真正的场分布数值

        return {"Ex": data_Ex[:, :].T, "Ey": data_Ey[:, :].T, "Ez": data_Ez[:, :].T,
                'f': f, 'wl': wl, 'x': xx, 'y': yy, 'z': zz}

    def swichout_del(self, del_name):
        self.fdtd.switchtolayout()
        self.fdtd.select(del_name)
        self.fdtd.delete()

    def quit(self):
        """
        关闭 lumerical
        """
        self.fdtd.close()

def get_params(N):
    period = np.random.random(N)*1.3 + 2
    h = np.random.random(N)*0.7 + 0.6
    w1 = period * (np.random.random(N)*0.06 + 0.2)
    w2 = period * (np.random.random(N)*0.06 + 0.2) + w1
    w3 = period * (np.random.random(N)*0.06 + 0.2) + w1
    l1 = period * (np.random.random(N)*0.06 + 0.2)
    l2 = period * (np.random.random(N)*0.06 + 0.2)
    l3 = period * (np.random.random(N)*0.06 + 0.2)

    par = np.stack([period, h, w1, w2, w3, l1, l2, l3], 1)
    print(par);print("A")
    params = np.round(par, 4)

    return params

if __name__ == '__main__':
    i = 1
    count = 2
    while i < count:
        print(i)

        params = get_params(1)                                   # 调用get_params函数，对设计参数随机采样
        par = params[0]
        np.savetxt(f"./chiral_data/params/{i}_par.txt", par)     # 保存采样参数为txt
        print(par)
        i += 1

if __name__ != '__main__':
    i = 1
    count = 10
    while i < count:
        print(i)

        params = get_params(1)                                   # 调用get_params函数，对设计参数随机采样
        par = params[0]
        np.savetxt(f"./chiral_data/params/{i}_par.txt", par)     # 保存采样参数为txt

        unity_sim = LumSimulation(hide=False)
        # ################## 参数定义 ##############
        # 基础结构
        unit = unity_sim.set_unit(unit=1e-6)
        period, h, w1, w2, w3, l1, l2, l3 = par * unit

        substrate_thickness = 9 * unit  # SiO2层衬底厚度
        source_z = h + 4 * unit         # 光源位置
        T_z = -3 * unit                 # 监视器位置

        wl_max = 5 * unit               # 设置仿真波长
        wl_min = 3 * unit
        nfreq = 201
        save_name = 'chiral_LCP'        # 仿真文件名，不用加后缀.fsp

        unity_sim.set_sim_paras(wl_max=wl_max, wl_min=wl_min, T_z=T_z, nfreq=nfreq, resolution=50)

        # # 材料定义
        unity_sim.define_material('SiO2', 1.45)   # 定义材料
        unity_sim.define_material('Si', 3.43)     # 定义材料
        unity_sim.define_material('air', 1, color=[0.501961, 0.501961, 1], opacity=0.65)

        # 添加 SiO2 衬底
        unity_sim.add_block(size=(2*period, 2*period, substrate_thickness), center=(0, 0, -substrate_thickness/2),
                            material_name='SiO2', name='substrate', color='Grey')

        # 创建超表面单元
        unity_sim.add_block(size=(w1, l1, h), center=(0, 0, h/2), material_name='Si', name='meta1', color='Red')
        unity_sim.add_block(size=(w2, l2, h), center=(-w2/2+w1/2, l1/2+l2/2, h/2), material_name='Si', name='meta2', color='Red')
        unity_sim.add_block(size=(w3, l3, h), center=(w3/2-w1/2, -l1/2-l3/2, h/2), material_name='Si', name='meta3', color='Red')

        # 添加源
        mode = 'LCP'
        unity_sim.add_source(size=(2*period, 2*period, 0), center=(0, 0, source_z), mode=mode)
        unity_sim.add_monitor(size=(2*period, 2*period, 0), center=(0, 0, T_z), name='T_monitor')

        # 添加 FDTD
        zmax = h + 4.5 * unit
        zmin = -3.5 * unit
        z_span = zmax - zmin
        z_cen = (zmin + zmax) / 2
        unity_sim.add_sim_area(size=(period, period, z_span), center=(0, 0, z_cen),
                               boundary='Periodic', mesh_accuracy=3, sim_time=5000e-15)

        # 仿真运行
        unity_sim.run(save_name=save_name)
        result_LCP = unity_sim.get_result_monitor(name='T_monitor')

        unity_sim.quit()

        # 结果处理 - 透射率
        wl = result_LCP["wl"]
        T_LCP = result_LCP["power"]

        fig, ax = plt.subplots(1, 1)
        plt.plot(wl, abs(T_LCP), 'b--', label='LCP')

        plt.xlabel("wavelength (μm)")
        plt.legend(loc="upper right")
        plt.savefig(f"chiral_data/LCP/LCP_{i}.png")
        plt.close()

        os.remove(f'{save_name}.fsp')
        os.remove(f'{save_name}_p0.log')

        np.savetxt(f"chiral_data/LCP/LCP_{i}.txt", abs(T_LCP))

        i += 1
        time.sleep(1)

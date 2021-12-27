import numpy as np
import sys
sys.path.append("../")
from data_structure.data_type import DisparityPlane,PVector3f
from cost.CostCompute import CostComputerPMS
import time

class PropagationPMS:
    def __init__(self, image_left, image_right, width, height, grad_left, grad_right,
                 plane_left, plane_right, config, cost_left, cost_right, disparity_map):
        self.image_left = image_left
        self.image_right = image_right
        self.width = width
        self.height = height
        self.grad_left = grad_left
        self.grad_right = grad_right
        self.plane_left = plane_left
        self.plane_right = plane_right
        self.config = config
        self.cost_left = cost_left
        self.cost_right = cost_right
        self.disparity_map = disparity_map
        self.cost_cpt_left = CostComputerPMS(image_left, image_right, grad_left, grad_right, width, height,
                                             config.patch_size, config.min_disparity, config.max_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.cost_cpt_right = CostComputerPMS(image_right, image_left, grad_right, grad_left, width, height,
                                             config.patch_size, -config.max_disparity, -config.min_disparity,
                                             config.gamma, config.alpha, config.tau_col, config.tau_grad)
        self.compute_cost_data()

    # 计算inital Cost 
    def compute_cost_data(self):
        for y in range(self.height):
            for x in range(self.width):
 
                p = self.plane_left[y, x]
                self.cost_left[y, x] = self.cost_cpt_left.compute_agg(x=x, y=y, p=p)
            if y%10==0:
                print("Finshed Inital Cost Aggregataion Rate :",y*1.0/self.height *100)
    
    # 进行空间传播
    def do_propagation(self, curr_iter):
        
        # 偶数次迭代从左上到右下传播
	    # 奇数次迭代从右下到左上传播
        direction = 1 if curr_iter % 2 == 0 else -1
        y = 0 if curr_iter % 2 == 0 else self.height - 1
        print(f"\r| Propagation iter {curr_iter}: 0", end="")
        
        for i in range(self.height):
            x = 0 if curr_iter % 2 == 0 else self.width - 1
            
            for j in range(self.width):
                # 本地空间传播
                self.spatial_propagation(x=x, y=y, direction=direction)
                
                # 平面优化   
                if not self.config.is_force_fpw:
                    self.plane_refine(x=x, y=y)
                
                # 视图传播
                self.view_propagation(x=x, y=y)
                x += direction
                
            y += direction
            
            if i%10==0:
                print("Iter: {} Propagation finshed Rate: ".format(curr_iter),i*1.0/self.height*100)
        
    
    # 本地空间传播
    def spatial_propagation(self, x, y, direction):
        '''
        x 为横坐标， y 为纵坐标，
        偶数次迭代从左上到右下传播
	    奇数次迭代从右下到左上传播
        '''


        # 获得当前的平面信息与cost
        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]

        xd = x - direction
        
        # 获取p左(右)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
        if 0 <= xd < self.width:
            plane = self.plane_left[y, xd]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost

        #获取p上(下)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
        yd = y - direction
        if 0 <= yd < self.height:
            plane = self.plane_left[yd, x]
            if plane != plane_p:
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane)
                if cost < cost_p:
                    plane_p = plane
                    cost_p = cost
        
        #更新 plane 和 Cost
        self.plane_left[y, x] = plane_p
        self.cost_left[y, x] = cost_p
    
    # 视图传播
    def view_propagation(self, x, y):
        # 获得当前的视差平面
        plane_p = self.plane_left[y, x]
        # 获得当前的disparity
        d_p = plane_p.to_disparity(x=x, y=y)
        
        # 找到右图我的对应的位置
        xr = int(x - d_p)
        if xr < 0 or xr > self.width - 1:
            return
        # 获得右图对应位置的视差平面和右图的损失
        plane_q = self.plane_right[y, xr]
        cost_q = self.cost_right[y, xr]
        
        # 找到左图对应dispariy 复制到右图对应的disparity

        plane_p2q = plane_p.to_another_view(x=x, y=y)

        # 此时右图的disparity
        d_q = plane_p2q.to_disparity(x=xr, y=y)
        cost = self.cost_cpt_right.compute_agg(x=xr, y=y, p=plane_p2q)
       
        # 如果合适那就更新
        if cost < cost_q:
            plane_q = plane_p2q
            cost_q = cost
        
        self.plane_right[y, xr] = plane_q
        self.cost_right[y, xr] = cost_q
    
    # 平面优化
    def plane_refine(self, x, y):
        
        # 获取视差的变换范围
        min_disp = self.config.min_disparity
        max_disp = self.config.max_disparity

        # 获得当前的位置的视差平面 plane_p 和对应的代价 cost
        plane_p = self.plane_left[y, x]
        cost_p = self.cost_left[y, x]
        # 获得当前的disparity
        d_p = plane_p.to_disparity(x=x, y=y)
        # 获得当前的surface normal
        norm_p = plane_p.to_norm()
        
        # 视差更新的范围的开始值： 1/2 disparity
        disp_update = (max_disp - min_disp) / 2.0
        norm_update = 1.0
        
        # 停止更新的大小
        stop_thres = 0.1

        # 传播到搜索范围小于0.1为止
        while disp_update > stop_thres:
            # delta disparity 
            disp_rd = np.random.uniform(-1.0, 1.0) * disp_update
            if self.config.is_integer_disparity:
                disp_rd = int(disp_rd)
            
            # 求得新的disparity
            d_p_new = d_p + disp_rd

            # 如果超过了范围，减小更新的步伐
            if d_p_new < min_disp or d_p_new > max_disp:
                disp_update /= 2
                norm_update /= 2
                continue

            # 在 -n max 到 n max里面sample
            if not self.config.is_force_fpw:
                norm_rd = PVector3f(
                    x=np.random.uniform(-1.0, 1.0) * norm_update,
                    y=np.random.uniform(-1.0, 1.0) * norm_update,
                    z=np.random.uniform(-1.0, 1.0) * norm_update,
                )
                # 保证最后一个维度不能为0， 因为分母不为0
                while norm_rd.z == 0.0:
                    norm_rd.z = np.random.uniform(-1.0, 1.0)
            else:
                norm_rd = PVector3f(x=0.0, y=0.0, z=0.0)
            
            # 求得normal 并且归一化
            norm_p_new = norm_p + norm_rd
            norm_p_new.normalize()

            # 计算在新 normal 和 disparity 下的 视差平面
            plane_new = DisparityPlane(x=x, y=y, d=d_p_new, n=norm_p_new)
            
            # 如果2者不相等
            if plane_new != plane_p:
                # 计算代价
                cost = self.cost_cpt_left.compute_agg(x=x, y=y, p=plane_new)
                # 如果新的代价比较小，更新 normal和disp
                if cost < cost_p:
                    plane_p = plane_new
                    cost_p = cost
                    d_p = d_p_new
                    norm_p = norm_p_new
                    self.plane_left[y, x] = plane_p
                    self.cost_left[y, x] = cost_p
            
            # 搜索范围减小 
            disp_update /= 2.0
            norm_update /= 2.0
 
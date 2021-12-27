import numpy as np
import cv2
from utils import PMSConfig
from data_structure.data_type import DisparityPlane,PVector3f
from propagation.propagation import PropagationPMS
import random

# Patch MatchStereo
class PatchMatchStereo:
    def __init__(self, width, height, config, random_seed=2022):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.image_left = None
        self.image_right = None
        self.width = width
        self.height = height
        self.config = config
        self.disparity_range = config.max_disparity - config.min_disparity
        self.gray_left = None
        self.gray_right = None
        self.grad_left = None
        self.grad_right = None
        self.cost_left = None
        self.cost_right = None
        self.disparity_left = None
        self.disparity_right = None
        self.plane_left = None
        self.plane_right = None
        self.mistakes_left = None
        self.mistakes_right = None
        self.invalid_disparity = 1024.0
        # Memory initialization
        self.Memory_Init(h=height, w=width)
        print("Memory Initalization is DONE!")
    
    # 初始化变量
    def Memory_Init(self, h, w):
        self.width = w
        self.height = h

        self.disparity_range = self.config.max_disparity - self.config.min_disparity
        self.gray_left = np.zeros([self.height, self.width], dtype=int)
        self.gray_right = np.zeros([self.height, self.width], dtype=int)
        self.grad_left = np.zeros([self.height, self.width, 2], dtype=float)
        self.grad_right = np.zeros([self.height, self.width, 2], dtype=float)
        self.cost_left = np.zeros([self.height, self.width], dtype=float)
        self.cost_right = np.zeros([self.height, self.width], dtype=float)
        self.disparity_left = np.zeros([self.height, self.width], dtype=float)
        self.disparity_right = np.zeros([self.height, self.width], dtype=float)
        self.plane_left = np.zeros([self.height, self.width], dtype=object)
        self.plane_right = np.zeros([self.height, self.width], dtype=object)
        self.mistakes_left = list()
        self.mistakes_right = list()

    # 视差，法向量，视差平面随机初始化
    def Random_Initalization(self):
        for y in range(self.height):
            for x in range(self.width):
                # random disparity
                disp_l = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                disp_r = np.random.uniform(float(self.config.min_disparity), float(self.config.max_disparity))
                if self.config.is_integer_disparity:
                    disp_l, disp_r = int(disp_l), int(disp_r)
                self.disparity_left[y, x], self.disparity_right[y, x] = disp_l, disp_r

                # random normal vector
                norm_l, norm_r = PVector3f(x=0.0, y=0.0, z=1.0), PVector3f(x=0.0, y=0.0, z=1.0)
                if not self.config.is_force_fpw:
                    norm_l = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    norm_r = PVector3f(
                        x=np.random.uniform(-1.0, 1.0),
                        y=np.random.uniform(-1.0, 1.0),
                        z=np.random.uniform(-1.0, 1.0),
                    )
                    while norm_l.z == 0.0:
                        norm_l.z = np.random.uniform(-1.0, 1.0)
                    while norm_r.z == 0.0:
                        norm_r.z = np.random.uniform(-1.0, 1.0)
                    norm_l, norm_r = norm_l.normalize(), norm_r.normalize()

                # random disparity plane
                # 左图
                self.plane_left[y, x] = DisparityPlane(x=x, y=y, d=disp_l, n=norm_l)
                # 右图
                self.plane_right[y, x] = DisparityPlane(x=x, y=y, d=disp_r, n=norm_r)

    # 从RBG中计算灰度值
    def Compute_Gray(self):
        for y in range(self.height):
            for x in range(self.width):
                b, g, r = self.image_left[y, x]
                self.gray_left[y, x] = int(r * 0.299 + g * 0.587 + b * 0.114)
                b, g, r = self.image_right[y, x]
                self.gray_right[y, x] = int(r * 0.299 + g * 0.587 + b * 0.114)
    
    # 从RBG中计算梯度
    def Compute_Gradient(self):
        for y in range(1, self.height - 1, 1):
            for x in range(1, self.width - 1, 1):
                # 使用Sobel算子 Left 梯度
                grad_x = self.gray_left[y - 1, x + 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y, x + 1] - 2 * self.gray_left[y, x - 1] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y + 1, x - 1]
                grad_y = self.gray_left[y + 1, x - 1] - self.gray_left[y - 1, x - 1] \
                         + 2 * self.gray_left[y + 1, x] - 2 * self.gray_left[y - 1, x] \
                         + self.gray_left[y + 1, x + 1] - self.gray_left[y - 1, x + 1]
                grad_y, grad_x = grad_y / 8, grad_x / 8
                self.grad_left[y, x, 0] = grad_x
                self.grad_left[y, x, 1] = grad_y

                # 使用Sobel算子 Right 梯度
                grad_x = self.gray_right[y - 1, x + 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y, x + 1] - 2 * self.gray_right[y, x - 1] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y + 1, x - 1]
                grad_y = self.gray_right[y + 1, x - 1] - self.gray_right[y - 1, x - 1] \
                         + 2 * self.gray_right[y + 1, x] - 2 * self.gray_right[y - 1, x] \
                         + self.gray_right[y + 1, x + 1] - self.gray_right[y - 1, x + 1]
                
                # 梯度/8 , 保证和RGB是同一个 Scale

                grad_y, grad_x = grad_y / 8, grad_x / 8
                self.grad_right[y, x, 0] = grad_x
                self.grad_right[y, x, 1] = grad_y

    def propagation(self,saved_intermediate=True):
        config_left = self.config.clone()
        config_right = self.config.clone()
        config_right.min_disparity = -1.0 * config_left.max_disparity
        config_right.max_disparity = -1.0 * config_left.min_disparity
        propa_left = PropagationPMS(self.image_left, self.image_right, self.width, self.height,
                                    self.grad_left, self.grad_right, self.plane_left, self.plane_right,
                                    config_left, self.cost_left, self.cost_right, self.disparity_left)
        propa_right = PropagationPMS(self.image_right, self.image_left, self.width, self.height,
                                    self.grad_right, self.grad_left, self.plane_right, self.plane_left,
                                    config_right, self.cost_right, self.cost_left, self.disparity_right)
        for it in range(self.config.n_iter):
            # 左图空间传播
            propa_left.do_propagation(curr_iter=it)
            # 右视图空间传播
            propa_right.do_propagation(curr_iter=it)

            # 保留中间产物
            if saved_intermediate:
                disp_L,disp_R,plane_L,plane_R = self.Plane_To_Disparity()
                np.save("results/dispL_{}.npy".format(it),disp_L)
                np.save("results/dispR_{}.npy".format(it),disp_R)
                np.save("results/planeL_{}.npy".format(it),plane_L)
                np.save("results/planeR_{}.npy".format(it),plane_R)                

    # 左右一致性行检查
    def lr_check(self):

        # 找出左图中的被遮挡的点
        for y in range(self.height):
            for x in range(self.width):
                # 得到左图的disparity
                disp = self.disparity_left[y, x]
                if disp == self.invalid_disparity:
                    self.mistakes_left.append([x, y])
                    continue
                # 对应的右图的 位置
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    # 找到右图同名点的disparity
                    disp_r = self.disparity_right[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        # 如果2者相差过大，那么有可能这个点是被遮挡的点
                        self.disparity_left[y, x] = self.invalid_disparity
                        self.mistakes_left.append([x, y])
                else:
                    self.disparity_left[y, x] = self.invalid_disparity
                    self.mistakes_left.append([x, y])

        # 找到右图中被遮挡的点
        for y in range(self.height):
            for x in range(self.width):
                disp = self.disparity_right[y, x]
                # 对于无效点直接跳过
                if disp == self.invalid_disparity:
                    self.mistakes_right.append([x, y])
                    continue
                col_right = round(x - disp)
                if 0 <= col_right < self.width:
                    disp_r = self.disparity_left[y, col_right]
                    if abs(disp + disp_r) > self.config.lr_check_threshold:
                        self.disparity_right[y, x] = self.invalid_disparity
                        self.mistakes_right.append([x, y])
                else:
                    self.disparity_right[y, x] = self.invalid_disparity
                    self.mistakes_right.append([x, y])

        # 空调填充
        #便是遍历每个无效像素（前面已经保存在mismatches数组里了），
        # 向左向右各搜寻到第一个有效像素，记录两个平面，
        # 再计算俩平面下的视差值，选较小值
    def fill_holes_in_disparity_map(self):
        for i in range(len(self.mistakes_left)):
            left_planes = list()
            x, y = self.mistakes_left[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_left[y, xs] != self.invalid_disparity:
                    left_planes.append(self.plane_left[y, xs])
                    break
                xs -= 1
            if len(left_planes) == 1:
                self.disparity_left[y, x] = left_planes[0].to_disparity(x=x, y=y)
            elif len(left_planes) > 1:
                d0 = left_planes[0].to_disparity(x=x, y=y)
                d1 = left_planes[1].to_disparity(x=x, y=y)
                self.disparity_left[y, x] = min(abs(d0), abs(d1))

        for i in range(len(self.mistakes_right)):
            right_planes = list()
            x, y = self.mistakes_right[i]
            xs = x + 1
            while xs < self.width:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs += 1
            xs = x - 1
            while xs >= 0:
                if self.disparity_right[y, xs] != self.invalid_disparity:
                    right_planes.append(self.plane_right[y, xs])
                    break
                xs -= 1
            if len(right_planes) == 1:
                self.disparity_right[y, x] = right_planes[0].to_disparity(x=x, y=y)
            elif len(right_planes) > 1:
                d0 = right_planes[0].to_disparity(x=x, y=y)
                d1 = right_planes[1].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = min(abs(d0), abs(d1))


    def Match(self, image_left, image_right):
        
        self.image_left, self.image_right = image_left, image_right
        
        # 随机初始化
        self.Random_Initalization()
        print("Random Initalization is Done!")
        
        # 计算灰度
        self.Compute_Gray()

        print("Compute Gray is Done!")
        
        # 计算灰度梯度
        self.Compute_Gradient()

        print("Compute Gradient is Done!")
        
        # 开始空间传播
        self.propagation()
        
        # 展示disparity图
        disp_L,disp_R,plane_L,plane_R = self.Plane_To_Disparity()

        # 左右一致性检查： 去掉小的被遮挡的区域
        if self.config.is_check_lr:
            self.lr_check()
        
        # 空洞填充
        if self.config.is_fill_holes:
            self.fill_holes_in_disparity_map()

        return disp_L,disp_R,plane_L,plane_R

    
    # 显示视差
    def Plane_To_Disparity(self):
        for y in range(self.height):
            for x in range(self.width):
                self.disparity_left[y, x] = self.plane_left[y, x].to_disparity(x=x, y=y)
                self.disparity_right[y, x] = self.plane_right[y, x].to_disparity(x=x, y=y)
        
        return self.disparity_left, self.disparity_right,self.plane_left,self.plane_right
        
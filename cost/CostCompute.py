from abc import abstractmethod
import numpy as np
import sys
sys.path.append("../")
from data_structure.data_type import DisparityPlane

COST_PINISH = 120.0


# 定义一个CostComputer 的通类
class CostComputer:
    def __init__(self, image_left, image_right, width, height, patch_size, min_disparity, max_disparity):
        self.image_left = image_left
        self.image_right = image_right
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.min_disparity = min_disparity
        self.max_disparity = max_disparity

    @staticmethod
    def fast_exp(v: float):
        v = 1 + v / 1024
        for _ in range(10):
            v *= v
        return v

    @abstractmethod
    def compute(self, x, y, d, *args, **kwargs):
        raise NotImplementedError("Compute should be implement.")


# 代价计算器
class CostComputerPMS(CostComputer):
    def __init__(self, image_left, image_right, grad_left, grad_right, width, height, patch_size, min_disparity,
                 max_disparity, gamma, alpha, tau_col, tau_grad):
        super(CostComputerPMS, self).__init__(image_left=image_left, image_right=image_right,
                                              width=width, height=height, patch_size=patch_size,
                                              min_disparity=min_disparity, max_disparity=max_disparity)
        self.grad_left = grad_left
        self.grad_right = grad_right
        self.gamma = gamma
        self.alpha = alpha
        self.tau_col = tau_col
        self.tau_grad = tau_grad

    def compute(self, x=0.0, y=0, d=0.0, col_p=None, grad_p=None):
        '''
        x 是横坐标， y 是纵坐标
        '''
        xr = x - d
        if xr < 0 or xr > self.width:
            return (1 - self.alpha) * self.tau_col + self.alpha * self.tau_grad

        # Color Space Distance
        col_p = col_p if col_p is not None else self.get_color(self.image_left, x=x, y=y)
        col_q = self.get_color(self.image_right, x=xr, y=y)
        dc = sum([abs(float(col_p[i]) - float(col_q[i])) for i in range(3)])
        dc = min(dc, self.tau_col)

        # Gradient Space Distance
        grad_p = grad_p if grad_p is not None else self.get_gradient(self.grad_left, x=x, y=y)
        grad_q = self.get_gradient(self.grad_right, x=xr, y=y)
        dg = abs(grad_p[0] - grad_q[0]) + abs(grad_p[1] - grad_q[1])
        dg = min(dg, self.tau_grad)

        return (1 - self.alpha) * dc + self.alpha * dg

    # 计算聚合代价： 一个patch_size的Local Cost
    def compute_agg(self, x, y, p: DisparityPlane):
        
        pat = self.patch_size // 2
        
        # 获得当前的RGB
        col_p = self.image_left[y, x, :]
        # 初始化 Cost
        cost = 0

        # 在patch-size内进行循环
        for r in range(-pat, pat, 1):
            y_ = y + r
            
            for c in range(-pat, pat, 1):
                x_ = x + c
                if y_ < 0 or y_ > self.height - 1 or x_ < 0 or x_ > self.width - 1:
                    continue
                # 获得Neighbor的disparity
                d = p.to_disparity(x=x, y=y)
                
                if d < self.min_disparity or d > self.max_disparity:
                    cost += COST_PINISH
                    continue

                # 当前邻居 的 RGB
                col_q = self.image_left[y_, x_, :]
                # Get color distance 
                dc = sum([abs(float(col_p[i]) - float(col_q[i])) for i in range(3)])
                # Get adaptive weight
                w = self.fast_exp(-dc / self.gamma)

                # Get the neighbour's gradient
                grad_q = self.grad_left[y_, x_]
                
                # add to the cost
                cost += w * self.compute(x=x_, y=y_, d=d, col_p=col_q, grad_p=grad_q)
        
        return cost

    def get_color(self, image, x: float, y: int):
        x1 = int(np.floor(x))
        x2 = int(np.ceil(x))
        ofs = x - x1
        color = list()
        for i in range(3):
            g1 = image[y, x1, i]
            g2 = image[y, x2, i] if x2 < self.width else g1
            color.append((1 - ofs) * g1 + ofs * g2)
        return color

    def get_gradient(self, gradient, x: float, y: int):
        x1 = int(np.floor(x))
        x2 = int(np.ceil(x))
        ofs = x - x1
        g1 = gradient[y, x1]
        g2 = gradient[y, x2] if x2 < self.width else g1
        x_ = (1 - ofs) * g1[0] + ofs * g2[0]
        y_ = (1 - ofs) * g1[1] + ofs * g2[1]
        return [x_, y_]
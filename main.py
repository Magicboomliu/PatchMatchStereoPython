import numpy as np
import cv2
from utils import PMSConfig
from PatchMatch import PatchMatchStereo
import matplotlib.pyplot as plt



if __name__=="__main__":

    # Read the Image data    
    left_image = cv2.imread("Data/Cone/im2.png")
    right_image = cv2.imread("Data/Cone/im6.png")

    height_, width_ = left_image.shape[0:2]

    # Get the PatchMatchStereo Configs
    config_ = PMSConfig("config/config.json")
    
    # 初始化函数
    pms = PatchMatchStereo(height=height_, width=width_, config=config_)
    
    # 开始传播优化
    dispL,dispR,plane_L,plane_R = pms.Match(image_left=left_image, image_right=right_image)

    plt.imshow(dispL,cmap='gray')
    plt.show()

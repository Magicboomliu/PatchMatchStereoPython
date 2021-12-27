import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_npy(file_path):
    data = np.load(file_path)
    return data

def MedianFilter(imarray,ksize=3):
    return cv2.medianBlur(imarray,ksize=3)

def BetterMedianFilter(imarray, k = 3, padding = None):
	height, width = imarray.shape
 
	if not padding:
		edge = int((k-1)/2)
		if height - 1 - edge <= edge or width - 1 - edge <= edge:
			print("The parameter k is to large.")
			return None
		new_arr = np.zeros((height, width))
		for i in range(height):
			for j in range(width):
				if i <= edge - 1 or i >= height - 1 - edge or j <= edge - 1 or j >= height - edge - 1:
					new_arr[i, j] = imarray[i, j]
				else:
					#nm:neighbour matrix
					nm = imarray[i - edge:i + edge + 1, j - edge:j + edge + 1]
					max = np.max(nm)
					min = np.min(nm)
					if imarray[i, j] == max or imarray[i, j] == min:
						new_arr[i, j] = np.median(nm)
					else:
						new_arr[i, j] = imarray[i, j]
		return new_arr
    




if __name__=="__main__":
    left_image = cv2.imread("../Data/Cone/im2.png")
    left_image = cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB)
    right_image = cv2.imread("../Data/Cone/im6.png")
    right_image = cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
    
    dispL = load_npy("dispL_2.npy")
    dispL = MedianFilter(dispL,5)

    dispR = load_npy("dispR_2.npy")

    # dispL = weightMedianFilter(dispL)
    # dispR = weightMedianFilter(-dispR)
    plt.subplot(1,2,1)
    plt.title("Left propagate thrid")
    plt.axis("off")
    plt.imshow(dispL,cmap='gray')
    plt.subplot(1,2,2)
    plt.title("Right propagate third")
    plt.axis("off")
    plt.imshow(-dispR,cmap='gray')
    plt.savefig("propagate_third.png")
    plt.show()

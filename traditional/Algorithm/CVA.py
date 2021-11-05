from PIL import Image
import sys
sys.path.append("../")
sys.path.append("../../")
from Threshold import *
from evalution import eval
from read_and_write import *

class CVA:
    def __init__(self,T1,T2,GT) :
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.GT = GT
        self.diff
        self.res
    def forward(self):
        self.diff = abs(self.T1-self.T2)
        self.res = K_means(self.diff)
        e = eval(self.res,GT=self.GT)
        return e
        
if __name__ =="__main__":
    path1 = 'D:\envi_img\Landsat\Taizhou\2000TM'
    path2 = 'D:\envi_img\Landsat\Taizhou\2003TM'

    data1 = gdal2np(path1)
    data2 = gdal2np(path2)
    GT = np.array(Image.open('D:\envi_img\Landsat\Taizhou\GT')) 

    cva = CVA(data1,data2,GT)
    cva.forward()
    print(cva.diff)
    pass
    
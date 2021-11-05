
import sys
sys.path.append("../")
sys.path.append("../../")
from Threshold import *
from evalution import eval
from read_and_write import *
import numpy as np
from PIL import Image

class PCA:
    '''
    Args:
        T1:     T1 Image
        T2:     T2 Image
        GT:     ground truth
        pcaChinel: transform bands
        diff:   diff Image
        res:    Change/Not change
    '''
      
    def __init__(self,T1,T2,GT) :
        self.T1 = T1
        self.T2 = T2
        self.GT = GT
        self.pcaChannel = 0
        self.ThresholdName = ''
        self.diff = None
        self.res = None
        

    def forward(self): 
        self.diff = self.pca(self.T1) - self.pca(self.T2)

        # 阈值方法选择

        if self.Threshold == "K_means":
            self.res = K_means(self.diff)
        if self.Threshold == "OTSU":
            self.res = OTSU(self.diff)



        pass
    
    # remember to use
    # choose chanel
    def set_Chanel(self,chanel):
        self.pcaChannel = chanel

        pass

    # choose Threshold algorithm
    def set_Threshold(self,threshold):
        self.ThresholdName = threshold
        pass

    def pca(self,T):
        H,W,C = T.shape
        x = T.reshape(H*W,C).T
        x = x-x.mean(axis=1).reshape(C,-1)
        covariance = np.cov(x,rowvar=True)
        eig_value,eig_vector = np.linalg.eig(covariance)
        eig_vector = eig_vector.T
        # sort eig_value,responding eig_vector
        eig_pair = []
        for i in range(len(eig_value)):
            eig_pair.append((eig_value[i],eig_vector[i]))
        # chose the component
        if self.pcaChannel < 1 and self.pcaChannel > 0:
            sum1 = 0
            for i in range(len(eig_value)):
                sum1 +=eig_value[i]
                if(sum1/sum(eig_value) >= self.pcaChanel):
                    break
            self.pcaChanel = i+1
        # chose numbers
        if self.pcaChannel == 0:
            self.pcaChannel = len(eig_value)
        
        matrix = np.zeros((self.pcaChannel,len(eig_vector[0])))
        for i in range(self.pcaChannel):
            matrix[i:] = eig_pair[i][1]


        return (matrix@x).T.reshape(H,W,-1)


if __name__ =="__main__":
    
    data1 = gdal2np('D:\envi_img\Landsat\Taizhou\\2000TM')
    data2 = gdal2np('D:\envi_img\Landsat\Taizhou\\2003TM')
    GT = np.array(Image.open('D:\envi_img\Landsat\Taizhou\\change.bmp'))
    pca1 = PCA(data1,data2,GT)
    res = pca1.pca(data1)
    np2gdal(res,'D:\envi_img\Landsat\Taizhou\\2003pca.tif')
    print(res.shape)
    pass
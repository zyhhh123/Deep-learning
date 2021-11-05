# gdal2numpy and numpy2g
import sys
import numpy as np
from osgeo import gdal,gdal_array
def gdal2np(path):
    # driver = gdal.GetDriverByName('HFA')
    # driver.Register()
    
    ds =gdal.Open(path,gdal.GA_ReadOnly)
    if ds is None:
        print('Could not open '+path)
        sys.exit(1)
    # H*W*C
    W = ds.RasterXSize
    H = ds.RasterYSize
    C = ds.RasterCount

    # get datatype
    datatype = ds.GetRasterBand(1).DataType
    # store to numpy
    data = np.zeros((H,W,C),dtype = gdal_array.GDALTypeCodeToNumericTypeCode(datatype))
    for i in range(C):
        band = ds.GetRasterBand(i+1)
        data[...,i] = band.ReadAsArray(0,0,W,H)
    return data
def np2gdal(data,path):
    driver = gdal.GetDriverByName('GTiff')

    H,W,C = data.shape
    ds = driver.Create(path,W,H,C,gdal.GDT_Float32)
    for i in range(C):
        ds.GetRasterBand(i+1).WriteArray(data[...,i])
import os      
if __name__ =='__main__':
    path = 'D:\\hello\\OCOD\\images\\Onera Satellite Change Detection dataset - Images\\rennes\\imgs_2_rect'
    file = os.listdir(path)
    
    for i in file:
        path1 = os.path.join(path,i)
        read = gdal2np(path1)
        print(read.shape)
  
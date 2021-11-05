import numpy as np
class eval:
    '''
    Args:
        res: result of threshold.   1 present change
                                    0 present unchanged
        BG:  ground truth
                1: change map
                2.change map,unchange map

        ConfusionMatrix: [
                            TP,FN
                            FP,TN
                         ]

    '''
    
    def __init__(self,res,*GT) :   
        self.res = res.ravel()
        if(len(GT) == 1):
            self.GT = GT.ravel()
        else:
            self.GT = [GT[0].ravel(),GT[1].ravel()]
        
        # get ConfusionMatrix
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        
        
        if(len(self.GT) ==1):
            for i in range(len(self.res)):
                if(self.res[i] !=0 and self.GT[i] !=0):
                    self.TP+=1
                elif(self.res[i] !=0 and self.GT[i] ==0):
                    self.FP+=1
                elif(self.res[i] ==0 and self.GT[i] ==0):
                    self.TN+=1      
        else:
            for i in range(len(self.res)):
                if(self.res[i] !=0 and self.GT[0][i] !=0):
                    self.TP+=1
                elif(self.res[i] !=0 and self.GT[1][i] !=0):
                    self.FP+=1
                elif(self.res[i] ==0 and self.GT[1][i] !=0):
                    self.TN+=1
                elif(self.res[i] ==0 and self[0][i] !=0):
                    self.FN+=1      
        self.Total = self.TN+self.TP+self.FN+self.FP

        pass
    def OA(self):
        return (self.TP+ self.TN)/self.Total
    # (po-pc)/(1-pc) 
    def Kappa(self):
        po = self.OA()
        pc = ((self.TN+self.FN)*(self.TN+self.FP)+
        (self.TP+self.FP)*(self.TP+self.FN))/self.Total**2
        return (po-pc)/(1-pc)
        pass
    
    def F1(self):
        pass
    def Recall(self):
        pass

    @staticmethod
    # compute ROC,AUC
    def ROC(GT,*diffs):

        
        pass
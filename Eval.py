import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        #print(self.pred)
        
    def Accuracy(self):
        #print(np.sum(self.pred))
        #print(np.sum(self.gold))
#        print(np.equal(self.pred, self.gold))
#        print(np.sum(np.equal(self.pred, self.gold)))
#        print(float(len(self.pred)))
#        print(float(len(self.gold)))
#        print(len((np.equal(self.pred, self.gold))))
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))

    def Recall(self):
#        return np.sum(np.logical_and((self.pred==1).all(), (self.gold==1).all())) /   \
#               (np.sum(np.logical_and((self.pred==1).all(), (self.gold==1).all())) +  \
#                np.sum(np.logical_and((self.pred==-1).all(), (self.gold==1).all())))  \
         #print(np.where(self.gold!=-1,self.gold,0))        
#         temp_test=np.where(self.gold!=-1,self.gold,0)
#        # print(np.sum(temp_test))
#         tp=np.sum(np.equal(self.pred, temp_test))
#         print(tp)
#         temp_pred=np.where(self.pred!=-1,self.pred,1)
#         tpn=np.sum(np.equal(temp_pred, self.gold))
#         print(tpn)
#         if tpn==0: return 0.0
#         else: return tp/tpn
         #return np.where(tpn!=0,tp/tpn,0)
#         print(np.sum(self.gold[np.where(self.pred==1)]==1))
#         print((np.sum(self.gold[np.where(self.pred==1)]==1)  \
#                                  + np.sum(self.gold[np.where(self.pred==-1)]==1)))
#         return np.sum(self.gold[np.where(self.pred==1)]==1)/  \
#                                 (np.sum(self.gold[np.where(self.pred==1)]==1)  \
#                                  + np.sum(self.gold[np.where(self.pred==-1)]==1))
        return recall_score(self.gold,self.pred)
    
    def Precision(self):
#        temp_test=np.where(self.gold!=-1,self.gold,0)
#        tp=np.sum(np.equal(self.pred, temp_test))
#        tpfp=np.count_nonzero(self.pred==1)
#        if tpfp==0: return 0.0
#        else: return tp/tpfp
         return precision_score(self.gold,self.pred)
     
    def PvRcurve(self):
        average_precision = average_precision_score(self.gold, self.pred)
        precision, recall, _ = precision_recall_curve(self.gold, self.pred)

        plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
        plt.show()
    
    
#a=np.array([1,-1,-1,1,-1,1])
#b=np.array([1,-1,-1,1,-1,1])
#sum([1 if i==1 else 0 for i in a])
#
##[1 if i == -1 else -1 for i in a]
#Y_pred=np.array([1,1,1,1])
#test=np.array([1,-1,-1,1])
#ev=Eval(Y_pred,test)
#print("For Positive Class:")
#print("Test Accuracy: ",ev.Accuracy())
#print("Test Recall: ",ev.Recall())
#print("Test Precision: ",ev.Precision())
#print("\n")
#print("For Negative Class:")
##print([1 if i == -1 else -1 for i in Y_pred])
##print("test:",[1 if i == -1 else -1 for i in test])
#ev_neg = Eval(np.array([1 if i == -1 else -1 for i in Y_pred]), np.array([1 if i == -1 else -1 for i in test]))
#print("Test Accuracy: ",ev_neg.Accuracy())
#print("Test Recall: ",ev_neg.Recall())
#print("Test Precision: ",ev_neg.Precision())



#np.equal(np.where(a==1),np.where(b==1))
#np.sum(b[np.where(a==1)]==1)
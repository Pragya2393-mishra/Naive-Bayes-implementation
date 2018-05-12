import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
         self.ALPHA = ALPHA
         self.data = data # training data
        #TODO: Initalize parameters
         self.vocab_len = 1
         self.count_positive = 1
         self.count_negative = 1
         self.num_positive_reviews = 1
         self.num_negative_reviews = 1
         self.total_positive_words = 1
         self.total_negative_words = 1
         self.P_positive = 1
         self.P_negative = 1
         self.deno_pos = 1
         self.deno_neg =1
         self.pos_words=[]
         self.neg_words=[]
         self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        #print(positive_indices)
        self.num_positive_reviews = sum([1 if i==1 else 0 for i in Y])
        self.num_negative_reviews = sum([1 if i==-1 else 0 for i in Y])
#        print(self.num_positive_reviews)
#        print(self.num_negative_reviews)
#        print(X)
#        print("assa")
#        print(np.sum(X[positive_indices,:]))
#        print(np.asarray(X[positive_indices,:]).reshape(1))
        self.count_positive = np.zeros(X.shape[1])
        self.count_negative = np.zeros(X.shape[1])
        #print("posi", self.count_positive)
        #print("nega",np.sum(self.count_negative ))
        self.total_positive_words = np.sum(X[positive_indices,:])
        self.total_negative_words = np.sum(X[negative_indices,:])
        
        self.deno_pos = self.total_positive_words + self.ALPHA*X.shape[1]
        self.deno_neg = self.total_negative_words + self.ALPHA*X.shape[1]
        
        # self.count_positive = 1
        # self.count_negative = 1
        rows,columns = X.nonzero()
#        print(X)
#        print(rows,columns)
#        print(self.count_positive[1])
#        print(self.count_negative)
        for i,j in zip(rows,columns):
            #print(X[i,j])
            #print("Y",self.data.Y[i])
           # print(j,"words in id order:",self.data.vocab.GetWord(j))
            if self.data.Y[i]==1:
              #  if self.count_positive[j] != 1:
                    self.count_positive[j]+=X[i,j]
            else:
               # if self.count_negative[j] != 1:
                    self.count_negative[j]+=X[i,j]
        #self.count_negative/=self.total_negative_words  
        #self.count_positive/=self.total_positive_words  
        self.count_positive = (self.count_positive + self.ALPHA)
        self.count_negative = (self.count_negative + self.ALPHA)        
        #print("pw:",self.count_positive)
        #print("npw:",(csr_matrix.sum(X[np.ix_(positive_indices)], axis=0)))
        #print("nw:",self.count_negative)
        #print("tp:",self.total_positive_words)

        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        #self.P_positive = self.num_positive_reviews/(self.num_positive_reviews + self.num_negative_reviews)
        self.P_positive = log(self.num_positive_reviews)-(log(self.num_positive_reviews)+log(self.num_negative_reviews))
        #self.P_negative = self.num_negative_reviews/(self.num_positive_reviews + self.num_negative_reviews)
        self.P_negative = log(self.num_negative_reviews)-(log(self.num_positive_reviews)+log(self.num_negative_reviews))
        pred_labels = []
        #print("py:",self.P_positive)
        w=X.shape[1]
        sh = X.shape[0]
        #print(X)
        for i in range(sh):

            #checks if the value of the data is zero or not if not then proceed

            z = X[i].nonzero()

            positive_sum = self.P_positive

            negative_sum = self.P_negative

            for j in range(len(z[0])):

                # Look at each feature

               

                row_index = i

                col_index = z[1][j]
         #       print("col:",col_index)
         #       print(self.count_positive[col_index])

                

                occurrence = X[row_index, col_index]

              

                P_pos = log(self.count_positive[col_index]) - log(self.deno_pos)

                positive_sum = positive_sum + occurrence * P_pos

               

                P_neg = log(self.count_negative[col_index]) - log(self.deno_neg)

                negative_sum = negative_sum + occurrence * P_neg

               
            print(positive_sum)
            print(positive_sum+negative_sum)
            if positive_sum > negative_sum:

                pred_labels.append(1.0)

            else:

                pred_labels.append(-1.0)

              

        return pred_labels


    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)   
        print(m)
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test,indexes):
         
         for i in indexes:

            # TO DO: Predict the probability of the i_th review in test being positive review

            # TO DO: Use the LogSum function to avoid underflow/overflow

            predicted_label = 0

            z = test.X[i].nonzero()

            positive_sum = self.P_positive

            negative_sum = self.P_negative

            for j in range(len(z[0])):

                row_index = i

                col_index = z[1][j]

              

                occurrence = test.X[row_index, col_index]

           

                P_pos = log(self.count_positive[col_index])

                positive_sum = positive_sum + occurrence * P_pos

               

                P_neg = log(self.count_negative[col_index])

                negative_sum = negative_sum + occurrence * P_neg

                

            predicted_prob_positive = exp(positive_sum - self.LogSum(positive_sum, negative_sum))

            predicted_prob_negative = exp(negative_sum - self.LogSum(positive_sum, negative_sum))

           

            if positive_sum > negative_sum:

                predicted_label=1.0

            else:

                predicted_label=-1.0

               

            #print (test.Y[i], test.X_reviews[i],predicted_label)

            # TO DO: Comment the line above, and uncomment the line below

            #print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative)

    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        print("For Positive Class:")
        print("Test Accuracy: ",ev.Accuracy())
        print("Test Recall: ",ev.Recall())
        print("Test Precision: ",ev.Precision())
        print("\n")
        print("For Negative Class:")
        ev_neg = Eval([1 if i == -1 else -1 for i in Y_pred], [1 if i == -1 else -1 for i in test.Y])
        print("Test Accuracy: ",ev_neg.Accuracy())
        print("Test Recall: ",ev_neg.Recall())
        print("Test Precision: ",ev_neg.Precision())
        ev.PvRcurve()

    def Features(self):
        pos_diff=np.zeros(self.data.X.shape[1])
        neg_diff=np.zeros(self.data.X.shape[1])
        for j in range(len(self.count_positive)):
                P_pos = log(self.count_positive[j]) - log(self.deno_pos)
                P_neg = log(self.count_negative[j]) - log(self.deno_neg)
                pos_diff[j]=(P_pos-P_neg)
                #neg_diff[j]=(P_neg-P_pos)*(self.count_negative[j]-self.count_positive[j])
        print("Top 20 Positive words with their weights:")        
        pos_index=pos_diff.argsort()[-20:][::-1]
        for j in pos_index:
            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
        
        print("Top 20 Negative words with their weights:")
        neg_index=pos_diff.argsort()[:20]
        for j in neg_index:
            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
        
#                pos_diff=np.zeros(self.data.X.shape[1])
#        neg_diff=np.zeros(self.data.X.shape[1])
#        for j in range(len(self.count_positive)):
#                P_pos = log(self.count_positive[j]) - log(self.deno_pos)
#                P_neg = log(self.count_negative[j]) - log(self.deno_neg)
#                pos_diff[j]=(P_pos-P_neg)*(self.count_positive[j]-self.count_negative[j])
#                neg_diff[j]=(P_neg-P_pos)*(self.count_negative[j]-self.count_positive[j])
#        print("Top 20 Positive words with their weights:")        
#        pos_index=pos_diff.argsort()[-20:][::-1]
#        for j in pos_index:
#            print("j:",self.data.vocab.GetWord(j)," ",pos_diff[j])
#        
#        print("Top 20 Negative words with their weights:")
#        neg_index=neg_diff.argsort()[:20]
#        for j in neg_index:
#            print("j:",self.data.vocab.GetWord(j)," ",neg_diff[j])
#        
#        neg=self.count_negative
#        neg_index=neg.argsort()[-10:][::-1]
#        for j in neg_index:
#            print(self.data.vocab.GetWord(j))
#        
    

if __name__ == "__main__":
    print(sys.argv)
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
#    print(traindata.X)
#    print(traindata.X_reviews)
#    print(traindata.Y)
#    print(traindata.vocab)
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)    
    
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    #print("tn:",nb.total_negative_words)
    print("Evaluating")
#    print("Test Accuracy: ", nb.Eval(testdata))
#    print("Evaluating")
#    print("Test Recall: ", nb.Eval1(testdata))
    nb.Eval(testdata)
   # nb.PredictProb(testdata,range(2))
    nb.Features()



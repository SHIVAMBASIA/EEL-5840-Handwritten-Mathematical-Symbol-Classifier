import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
import time
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')

######################### Inputs ##################################

# The inputs can be passed as npy files here
images = np.load(sys.argv[1])           # data set X.       Expected dimensions(X,300,300)
labels = np.load(sys.argv[2])        # desired output y. Expected dimensions(y,)

########################## Load saved model ####################################
saved_model = "saved_model.pt"

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 11, 3, 1)
        self.conv2 = nn.Conv2d(11, 20, 3, 1)
        self.conv3 = nn.Conv2d(20, 30, 3, 1)
        self.fc1 = nn.Linear(17*17*30, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 25)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.avg_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.avg_pool2d(X, 2)
        X = F.relu(self.conv3(X))
        X = F.avg_pool2d(X, 2)
        X = X.view(-1, 17*17*30)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
torch.manual_seed(101)

CNNmodel = ConvolutionalNetwork()#.to(device)
CNNmodel.load_state_dict(torch.load(saved_model))
CNNmodel.eval()


########################## Test function definition ####################################

def normalize_data(X):
    mu = np.mean(X)
    std = np.std(X)
    return ((X-mu)/std)

# Tests the model for a given set of images, labels and prints the accuracy and confusion matrix
# returns a list of predicted labels for each X
def test(X_test_combined, y_test_combined):
    
    # Set parameters
    batch_size = 1
    
    X_test_combined = 255 - X_test_combined
    X_test_combined=X_test_combined.T
    X_test=[]
    equation_length=[]
    for i in range(X_test_combined.shape[0]):
        img=X_test_combined[i,:].reshape(300,300)
        if img is not None:
            if img is not None:
                #img=~img
                ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
                ctrs,ret=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
                w=int(150)
                h=int(150)
                train_data=[]
                rects=[]
                for c in cnt :
                    x,y,w,h= cv2.boundingRect(c)
                    rect=[x,y,w,h]
                    rects.append(rect)
                bool_rect=[]
                for r in rects:
                    l=[]
                    for rec in rects:
                        flag=0
                        if rec!=r:
                            if r[0]<(rec[0]+rec[2]+10) and rec[0]<(r[0]+r[2]+10) and r[1]<(rec[1]+rec[3]+10) and rec[1]<(r[1]+r[3]+10):
                                flag=1
                            l.append(flag)
                        if rec==r:
                            l.append(0)
                    bool_rect.append(l)
                dump_rect=[]
                for i in range(0,len(cnt)):
                    for j in range(0,len(cnt)):
                        if bool_rect[i][j]==1:
                            area1=rects[i][2]*rects[i][3]
                            area2=rects[j][2]*rects[j][3]
                            if(area1==min(area1,area2)):
                                dump_rect.append(rects[i]) 
                final_rect=[i for i in rects if i not in dump_rect]
                for r in final_rect:
                    x=r[0]
                    y=r[1]
                    w=r[2]
                    h=r[3]
                    im_crop =thresh[y:y+h+10,x:x+w+10]
                    im_resize = cv2.resize(im_crop,(150,150))
                    X_test.append(im_resize)
                equation_length.append(len(final_rect))
    X_test=np.array(X_test)
    X_test = X_test.reshape(len(X_test),1,150,150)
    # Normalization of data
    X_test = normalize_data(X_test)
    y_test=[]
    actual_equation_length=[]
    for i in y_test_combined:
        for j in str(i):
            if j=='u':
                j=='10'
            y_test.append(int(j))
        actual_equation_length.append(len(str(i)))
    for i in range(len(X_test)-len(y_test)):
        y_test.append(10)
    y_test=np.array(y_test)
    
    # Converting input numpy array to pytorch tensors
    X_testTensor = torch.Tensor(X_test)
    y_testTensor = torch.Tensor(y_test)
    
    y_testTensor = y_testTensor.type(torch.LongTensor)
    
    # Create test dataset and loader
    test_data = TensorDataset(X_testTensor,y_testTensor) 
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True) 
    
    # Testing start time
    start_time = time.time()
    
    test_correct = 0
    final_predicted = []
    final_actual = []

    # For each test batch
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            # Apply the model
            y_pred = CNNmodel(X_test)

            # Tally the number of correct predictions          
            predicted_value = torch.max(y_pred.data, 1)[0]
            if abs(predicted_value.data[0].item())>=abs(1e-7) or abs(predicted_value.data[0].item())==0 :
                predicted = torch.max(y_pred.data, 1)[1]
            else:
                predicted=torch.tensor([10])
            test_correct += (predicted == y_test).sum()
            
            final_predicted += predicted
            final_actual += y_test
    k=0
    final_predicted_equation=[]
    final_actual_equation=[]
    print(f'Accuracy: {test_correct*(100/len(test_data))}')

    print(confusion_matrix(final_actual,final_predicted))
    print(classification_report(final_actual,final_predicted))   
    print(f'\nDuration for testing: {time.time() - start_time:.0f} seconds') # print the time elapsed for testing
############################## Run test with data ############################
    for i in equation_length:
        s=''
        a=''
        for j in range(i):
            value_s=str(final_predicted[k+j]).split('(')[1].split(')')[00]
            value_a=str(final_actual[k+j]).split('(')[1].split(')')[00]
            if value_s=='10':
                value_s='u'
            if value_a=='10':
                value_a='u'
            s+=value_s
            a+=value_a

        final_predicted_equation.append(s)
        final_actual_equation.append(a)
        k+=i
    df = pd.DataFrame({'final_actual_equation':final_actual_equation,'final_predicted_equation':final_predicted_equation})
    df.to_csv("output_equation.csv", index=False)
    df2 = pd.DataFrame({'actual_equation_length':actual_equation_length,'predicted_equation_length':equation_length})
    df2.to_csv("output_equation_length.csv", index=False)
test(images, labels)


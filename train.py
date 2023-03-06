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
import warnings
warnings.simplefilter(action='ignore')

######################### inputs ##################################

# The inputs can be passed as npy files here

images = np.load(sys.argv[1])         # data set X.        Expected dimensions(X,300,300)
labels = np.load(sys.argv[2])        # desired output y.  Expected dimensions(y,)
labels[labels == -1] = 10

########################## Model ####################################

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
criterion = nn.CrossEntropyLoss()#.to(device)
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)


########################## function definition ####################################

def normalize_data(X):
    mu = np.mean(X)
    std = np.std(X)
    return ((X-mu)/std)

# Trains the model for input images and labels. Prints the accuracy on the training and validation datasets. Also prints the final confusion matrix.
def train(images, labels):
    #images, labels = images.to(device), labels.to(device)
    # Set parameters
    epochs = 10
    
    batch_size = 10
    max_train_batches = 3000
    max_val_batches = 3000
    
    images = 255 - images

    images=images.T
    #X = images.T
    X_train_resized = np.array([cv2.resize(x.reshape(300,300),(150,150)).reshape(150*150) for x in images])
    #X_train_resized.shape
    X_train_resized = X_train_resized.reshape(len(X_train_resized),1,150,150)
    print(X_train_resized.shape)
    #X_train_resized, labels = X_train_resized.to(device), labels.to(device)
    
    #Set test_size = 0, if validation set is not needed.
    X_train, X_val, y_train, y_val = train_test_split(X_train_resized, labels, test_size=0.20, random_state=42)
    
    
    # Normalization of data
    X_train = normalize_data(X_train)
    X_val = normalize_data(X_val)
    
    
    # Converting input numpy array to pytorch tensors
    X_trainTensor = torch.Tensor(X_train)
    X_valTensor = torch.Tensor(X_val)
    y_trainTensor = torch.Tensor(y_train)
    y_valTensor = torch.Tensor(y_val)
    
    y_trainTensor = y_trainTensor.type(torch.LongTensor)
    y_valTensor = y_valTensor.type(torch.LongTensor)
    
    
    # Create training dataset and loader
    train_data = TensorDataset(X_trainTensor,y_trainTensor) 
    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True) 
    
    # Create validation dataset and loader
    val_data = TensorDataset(X_valTensor,y_valTensor) 
    val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=True) 
    
    
    print(f'Training images available: {len(train_data)}')
    print(f'Validation images available:  {len(val_data)}')

    
    # Training start time
    start_time = time.time()
    
    
    train_losses = []
    val_losses = []
    train_correct = []
    val_correct = []
    final_predicted = []
    final_actual = []
    
    # For each Epoch
    for i in range(epochs):
        trn_corr = 0
        val_corr = 0
        
        # For each training batch
        for b, (X_train, y_train) in enumerate(train_loader):
            
            # Limit the number of batches
            if b == max_train_batches:
                break
            b += 1
            
            # Apply the model
            y_pred = CNNmodel(X_train)
            loss = criterion(y_pred, y_train)
     
            # Tally the number of correct predictions
            predicted = torch.max(y_pred.data, 1)[1]
            trn_corr += (predicted == y_train).sum()
            
            # Store predictions of final epoch for confusion matrix
            if(i == epochs-1):
                final_predicted += predicted
                final_actual += y_train
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # Print interim results
            if b%100 == 0:
                print(f'epoch: {i:2}  batch: {b:4}  loss: {loss.item():10.8f} accuracy: {trn_corr.item()*100/(10*b):7.3f}%')
    
        train_losses.append(loss)
        train_correct.append(trn_corr)
    
        # For each validation batch
        with torch.no_grad():
            for b, (X_val, y_val) in enumerate(val_loader):
                
                # Limit the number of batches
                if b == max_val_batches:
                    break
    
                # Apply the model
                y_pred = CNNmodel(X_val)
    
                # Tally the number of correct predictions
                predicted = torch.max(y_pred.data, 1)[1]
                val_corr += (predicted == y_val).sum()
                
                # Store predictions of final epoch for confusion matrix
                if(i == epochs - 1):
                    final_predicted += predicted
                    final_actual += y_val
    
        loss = criterion(y_pred, y_val)
        val_losses.append(loss)
        val_correct.append(val_corr)
        
        
        print(f'Overall train accuracy after {i}th epoch: {train_correct[i]*(100/len(train_data))}')
        print(f'Overall validation accuracy after {i}th epoch: {val_correct[i]*(100/len(val_data))}')
    print(confusion_matrix(final_actual, final_predicted))
    print(classification_report(final_actual,final_predicted))
    print(f'\nDuration for training: {time.time() - start_time:.0f} seconds') # print the time elapsed for training
    
############################## Train with data and save model ############################

# Train with data
train(images, labels)

# Save the model
torch.save(CNNmodel.state_dict(), "saved_model.pt")



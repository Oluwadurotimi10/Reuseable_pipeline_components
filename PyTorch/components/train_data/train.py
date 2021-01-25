#importing libraries
import argparse

def train(X_train, y_train, train_loader):
    #importing libraries
    import numpy as np
    import joblib
    import torch
    import torch.nn as nn
    import torch.optim as optim

   #loading the inputs
    X_train = np.load(X_train)
    y_train = np.load(y_train)
    #deserialize data from output directory
    train_loader = joblib.load(train_loader)
    #setting model hyper-parameters
    EPOCHS = 150
    BATCH_SIZE =10
    LEARNING_RATE = 0.001
    #defining neural network architecture
    class binaryClassification(nn.Module):
        def __init__(self):
            super(binaryClassification, self).__init__()
            #number of input features is 12
            self.layer_1 = nn.Linear(12, 16)
            self.layer_2 = nn.Linear(16, 8)
            self.layer_out = nn.Linear(8, 1) 
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1)
            self.batchnorm1 = nn.BatchNorm1d(16)
            self.batchnorm2 = nn.BatchNorm1d(8)       
        #feed forward network
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.layer_out(x)
            return x
    #initializing optimizer and loss
    classifier = binaryClassification()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr = LEARNING_RATE)

    #function to calculate accuracy
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        results_sum = (y_pred_tag == y_test).sum().float()
        acc = results_sum/y_test.shape[0]
        acc =torch.round(acc*100)
        return acc
    #training the model
    classifier.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            #setting gradient to 0 per mini-batch
            optimizer.zero_grad()
            y_pred = classifier(X_batch)
            loss =criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            print(f'Epoch {e+0:03}: | Loss:{epoch_loss/len(train_loader):.5f} | Acc:{epoch_acc/len(train_loader):.3f}')
    #saving model
    torch.save(classifier.state_dict(),'pyclassifier.pth')
    
#defining and parsing arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--train_loader')
    args = parser.parse_args()
    print('Done with training')
    train(args.X_train,args.y_train, args.train_loader)

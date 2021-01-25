#importing libraries
import argparse

def dataloader(X_train,y_train,X_test):
    #importing libraries
    import joblib
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

    #loading the inputs
    X_train = np.load(X_train)
    X_test = np.load(X_test)
    y_train = np.load(y_train)

    #creating custom dataset for loading
    #train data
    class trainData(Dataset):
        def __init__(self, X_data, y_data):
            self.X_data = X_data
            self.y_data = y_data
            
        def __getitem__(self,index):
            return self.X_data[index], self.y_data[index]
        
        def __len__(self):
            return len(self.X_data)
    train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train.values))
    #test data
    class testData(Dataset):
        def __init__(self, X_data):
            self.X_data = X_data
            
        def __getitem__(self,index):
            return self.X_data[index]
        
        def __len__(self):
            return len(self.X_data)
    test_data = testData(torch.FloatTensor(X_test))
    #model hyperparameter
    BATCH_SIZE =10
    #defining dataloader to load data in batches
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)  
    #saving the dataloaders
    joblib.dump(train_loader,'train_loader')
    joblib.dump(test_loader,'test_loader')

#defining and parsing arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--X_test')
    args = parser.parse_args()
    print('Done with loading data')
    dataloader(args.X_train,args.y_train,args.X_test)
         
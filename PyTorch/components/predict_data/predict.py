#importing libraries
import argparse

def predict(y_test,model,test_loader):
    #importing libraries
    import joblib
    import numpy as np
    import pandas as pd
    import torch

    #loading the model and inputs
    y_test = np.load(y_test)
    classifier = classifier.load_state_dict(torch.load(model))
    test_loader = joblib.load(test_loader)

    #function to calculate accuracy
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        results_sum = (y_pred_tag == y_test).sum().float()
        acc = results_sum/y_test.shape[0]
        acc =torch.round(acc*100)
        return acc

    #test model
    y_pred_list = []
    classifier.eval()
    #ensures no back propagation during testing and reduces memeory usage
    with torch.no_grad():
        for X_batch in test_loader:
            y_test_pred = classifier(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred) 
            y_pred_list.append(y_pred_tag.cpu().numpy())
        y_pred_list = [i.squeeze().tolist() for i in y_pred_list]
        #accuracy
        acc = binary_acc(y_pred_list, y_test)
        #print(acc)
    #Serialize the output
    #saving prediction as csv
    df1 = pd.DataFrame(y_test)
    df1.reset_index(inplace=True)
    df1.drop(columns=['index'], axis=1, inplace=True)
    df2 = pd.DataFrame(y_pred_list)
    df = pd.concat([df1, df2], axis=1)
    df.columns=['target','predicted']
    df.to_csv('part-result_pytorch.csv')

#defining and parsing arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    parser.add_argument('--test_loader')
    args = parser.parse_args()
    print('Prediction has be saved successfully!')
    predict(args.y_test, args.model, args.test_loader)
    
#importing libraries
import argparse

def predict(X_test,y_test,model):
    #importing libraries
    import joblib
    import numpy as np
    import pandas as pd
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from sklearn.metrics import confusion_matrix
    #loading the model and inputs
    X_test = np.load(X_test)
    y_test = np.load(y_test)
    classifier = load_model(model)

    #Evaluate the model and print the results
    test_loss, test_acc = classifier.evaluate(X_test,  y_test, verbose=0)
    
    #model's prediction on test data
    y_pred = classifier.predict(X_test)
    # create a threshold for the confution matrics
    y_pred=(y_pred>0.5)

    #saving pred and actual as csv file
    vocab = [0,1]
    cm = confusion_matrix(y_test, y_pred, labels=vocab)
    cm_data = []
    for target_index,target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            cm_data.append((vocab[target_index], vocab[predicted_index], count))

    cm_df = pd.DataFrame(cm_data, columns=['target','predicted','count'])
    
    #serialize data to be used for confusion matrix
    joblib.dump(cm_df, 'matrix_data')
    #saving the test_loss and test_acc
    joblib.dump(test_loss, 'test_loss')
    joblib.dump(test_acc, 'test_acc')
    
#defining and parsing arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    print('Prediction has be saved successfully!')
    predict(args.X_test, args.y_test, args.model)
    
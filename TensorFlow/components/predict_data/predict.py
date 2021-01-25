#importing libraries
import argparse

def predict(X_test,y_test,model):
    #importing libraries
    import joblib
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    #loading the model and inputs
    X_test = np.load(X_test)
    y_test = np.load(y_test)
    classifier = load_model(model)

    #Evaluate the model and print the results
    test_loss, test_acc = classifier.evaluate(X_test,  y_test, verbose=0)
    #print('Test accuracy:', test_acc)
    #print('Test loss:', test_loss)
    #model's prediction on test data
    y_pred = classifier.predict(X_test)
    # create a threshold for the confution matrics
    y_pred=(y_pred>0.5)
    #Serialize the output
    joblib.dump((test_acc, test_loss, y_test, y_pred),'results')
    
#defining and parsing arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    parser.add_argument('--model')
    args = parser.parse_args()
    print('Prediction has be saved successfully!')
    predict(args.X_test, args.y_test, args.model)
    
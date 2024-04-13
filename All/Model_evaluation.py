import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def model_evaluation(model, X_train, X_test, y_train, y_test, threshold = 0.5, evaluate = False, reshape_outcome = False):
    print("Model Evaluation:")
    pred_proba_train = model.predict(X_train)
    pred_train = tf.greater(pred_proba_train, threshold)
    pred_train = outcome_reshape(pred_train)
    if reshape_outcome == True:
        y_train = outcome_reshape(y_train)
    tn, fp, fn, tp = confusion_matrix(y_train, pred_train, labels = [0,1]).ravel()
    
    if evaluate == True:
        score = model.evaluate(X_train, y_train, verbose=0)
        print("Train set:")
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])
        print("Sensitivity:", tp/(tp+fn), "| Specificity:", tn/(tn+fp))
        ConfusionMatrixDisplay.from_predictions(y_train, pred_train)
    else:
        print("Train set:")
        print("Train Accuracy:", (tp+tn)/(tp+tn+fp+fn), "| Sensitivity:", tp/(tp+fn), "| Specificity:", tn/(tn+fp))
        ConfusionMatrixDisplay.from_predictions(y_train, pred_train)
    print("\n")

    

    pred_proba_test = model.predict(X_test)
    pred_test = tf.greater(pred_proba_test, threshold)
    pred_test = outcome_reshape(pred_test)
    if reshape_outcome == True:
        y_test = outcome_reshape(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels = [0,1]).ravel()
    if evaluate == True:
        print("Test set:")
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print("Sensitivity:", tp/(tp+fn), "| Specificity:", tn/(tn+fp))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_test)
    else:
        print("Test set:")
        #print("tn:", tn, "fp:", fp, "fn:", fn, "tp:", tp )
        print("Accuracy: ", (tp+tn)/(tp+tn+fp+fn), "| Sensitivity:", tp/(tp+fn), "| Specificity:", tn/(tn+fp))
        ConfusionMatrixDisplay.from_predictions(y_test, pred_test)
        

def outcome_reshape(input):
    out = []
    for i in range(len(input)):
        if input[i][0] == True: #normal is true
            out.append(0) #0
        else:
            out.append(1)
    return out

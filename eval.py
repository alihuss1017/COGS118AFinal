from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def eval_models(lr_model, svm_model, rf_model, X, Y):
    _, ax = plt.subplots(2,1, figsize = (10, 10))
    lr_pred = lr_model.predict(X)
    svm_pred = svm_model.predict(X)
    rf_pred = rf_model.predict(X)

    lr_acc = accuracy_score(Y, lr_pred)
    svm_acc = accuracy_score(Y, svm_pred)
    rf_acc = accuracy_score(Y, rf_pred)

    lr_f1 = f1_score(Y, lr_pred)
    svm_f1 = f1_score(Y, svm_pred)
    rf_f1 = f1_score(Y, rf_pred)

    print(f"Accuracies:\nlogistic regression: {lr_acc}\n SVM: {svm_acc} Random Forests: {rf_acc}")

    print(f"f1Scores:\nlogistic regression: {lr_f1}\n SVM: {svm_f1} Random Forests: {rf_f1}")

    ax[0].bar(['Log Reg', 'SVM', 'RFs'], [lr_acc, svm_acc, rf_acc])
    ax[0].set_title("Testing Accuracies")
    ax[0].set_xlabel("Model")
    ax[0].set_ylabel("Accuracy")
    ax[1].bar(['Log Reg', 'SVM', 'RFs'], [lr_f1, svm_f1, rf_f1])
    ax[1].set_title("Testing f1Scores")
    ax[1].set_xlabel("Model")
    ax[1].set_ylabel("f1Score")
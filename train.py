from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import copy

def train_models(X, Y, models, size):
    f1_scores = []
    train_accs = []
    val_accs = []
    best_acc = 0
    for model_inst in models:

        train_acc = 0
        val_acc = 0
        f1 = 0
        for i in range(3):
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = size)
            model = copy.deepcopy(model_inst)
            model.fit(X_train, Y_train)

            Y_train_hat = model.predict(X_train)
            Y_val_hat = model.predict(X_val)

            train_acc += accuracy_score(Y_train, Y_train_hat)
            val_acc += accuracy_score(Y_val, Y_val_hat)
            f1 += f1_score(Y_val, Y_val_hat)

        f1_scores.append(f1/3)
        train_accs.append(train_acc / 3)
        val_accs.append(val_acc / 3)

        if best_acc < (val_acc / 3):
            best_acc = val_acc / 3
            best_model = model
    return train_accs, val_accs, f1_scores, best_acc, best_model
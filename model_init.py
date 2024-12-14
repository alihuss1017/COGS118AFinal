from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def model_list():
    svms = []
    logregs = []
    rfs = []
    splits = [1,2,4,8,12,16,20]
    ridge = 10e-9

    while (True):
        if ridge > 10e3:
            break
        logregs.append(LogisticRegression(C = ridge))
        ridge *= 10

    svms.append(SVC(kernel = 'linear'))
    svms.append(SVC(kernel = 'poly', degree = 2))
    svms.append(SVC(kernel = 'poly', degree = 3))
    svms.append(SVC(kernel = 'rbf', gamma = .001))
    svms.append(SVC(kernel = 'rbf', gamma = .05))
    svms.append(SVC(kernel = 'rbf', gamma = 1))

    for split in splits:
        rfs.append(RandomForestClassifier(n_estimators = 1024, max_features = split))

    return logregs, svms, rfs
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer, confusion_matrix, fbeta_score
import numpy as np

def precision_thresh(predict_probs, y_test, thresh = 0.5):
    y_train_pred_prob = model.predict_proba(X_train)
    y_test_pred_prob = model.predict_proba(X_test)
    
    y_train_pred = np.where(y_train_pred_prob >= thresh, 1, 0)
    y_test_pred = np.where(y_test_pred_prob >= thresh, 1, 0)
    
    precision_train=precision_score(y_train, y_train_pred, pos_label=1)
    precision_test =precision_score(y_test, y_test_pred, pos_label=1)
    
    return precision_train, precision_test    


def recall_thresh(model, X_train, y_train, X_test, y_test,  predict_probs, y_test, thresh = 0.5):
    y_train_pred_prob = model.predict_proba(X_train)
    y_test_pred_prob = model.predict_proba(X_test)
    
    y_train_pred = np.where(y_train_pred_prob >= thresh, 1, 0)
    y_test_pred = np.where(y_test_pred_prob >= thresh, 1, 0)
    
    recall_train=recall_score(y_train, y_train_pred, pos_label=1)
    recall_test =recall_score(y_test, y_test_pred, pos_label=1)
    
    return recall_train, recall_test


def f1_score(model, X_train, y_train, X_test, y_test)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    f1_train = f1_score(y_train, y_train_pred, average='micro')
    f1_test = f1_score(y_test, y_test_pred, average='micro')
    return f1_train, f1_test


def calc_predict_proba(model, X_train, y_train, X_test, y_test):
    y_train_pred_prob = knn.predict_proba(X_train)
    y_test_pred_prob = knn.predict_proba(X_test)
    return y_train_pred_prob, y_test_pred_prob


def calc_scores(y_train,y_train_pred_prob, y_test,y_test_pred_prob, threshold=0.5):
    y_train_pred = np.where(y_train_pred_prob > threshold, 1, 0)
    y_test_pred = np.where(y_test_pred_prob > threshold, 1, 0)
    
    train_acc = knn.score(X_train, y_train)
    test_acc = knn.score(X_test, y_test)

    rs = recall_score(y_test, y_test_pred[:, 1], pos_label=1)
    ps = precision_score(y_test, y_test_pred[:, 1], pos_label=1)
    f1 = f1_score(y_test, y_test_pred[:, 1], pos_label=1, average='micro')
    ac = accuracy_score(y_test, y_test_pred[:, 1])
    return train_acc, test_acc, rs, ps, f1, ac, y_test_pred


def confusion_matrix_display(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred[:, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    # disp.plot()
    # plt.show()
    TN, FP, FN, TP = cm.ravel()
    return TN, FP, FN, TP

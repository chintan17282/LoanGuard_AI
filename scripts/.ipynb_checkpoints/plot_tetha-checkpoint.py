import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, make_scorer, confusion_matrix, fbeta_score

def get_predictions(cv, X_test, y_test): 
    accuracy, precision, recall, specificity, f1 = [],[],[],[],[]
    for r in np.linspace(0.3, 0.9, 13):
        low_preds = np.where(cv.predict_proba(X_test)[:, 1] > r, 'yes', 'no')
        accuracy.append(accuracy_score(y_test, low_preds))
        precision.append(precision_score(y_test, low_preds, pos_label='yes'))
        recall.append(recall_score(y_test, low_preds, pos_label='yes'))
        specificity.append(recall_score(y_test, low_preds, pos_label='no'))
        f1.append(f1_score(y_test, low_preds, pos_label='yes'))
    return accuracy, precision, recall, specificity, f1

def plot_model(cv, X_test, y_test):
    accuracy, precision, recall, specificity, f1 = get_predictions(cv, X_test, y_test)
    plt.figure(figsize=(20, 8))
    plt.plot(np.linspace(0.3, 0.9, 13), accuracy, label='Accuracy', marker='o')
    plt.plot(np.linspace(0.3, 0.9, 13), precision, label='Precision', marker='o')
    plt.plot(np.linspace(0.3, 0.9, 13), recall, label='Recall', marker='o')
    plt.plot(np.linspace(0.3, 0.9, 13), specificity, label='Specificity', marker='o')
    plt.plot(np.linspace(0.3, 0.9, 13), f1, label='F1', marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold to Score')
    plt.legend()
    plt.grid(True)
    plt.show()
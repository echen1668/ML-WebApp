import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import auc
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from skopt import BayesSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
#from scipy import interp
from scipy.stats import norm
import openpyxl 
from openpyxl import load_workbook
import xlsxwriter
import magic 
import random 
from random import randint
from random import uniform
from scipy import stats
import mrmr
from mrmr import mrmr_classif
import xgboost
import catboost
import shap
from scipy import stats
import os
import joblib as joblib
from joblib import dump, load
import json
import tkinter as tk
from tkinter import *
#np.random.seed(1000)
rstate = 12

# Import common tools
import Common_Tools

# define a model to use and its set of parameters
def get_classifier(alg, param_vals="None"):
    print(param_vals)
    est_rs = 1000
    if alg == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(random_state=est_rs)
        if param_vals == 'None':
            param_vals = {'max_depth': list(np.arange(2, 14, 1)),
                          'n_estimators': list(np.arange(5, 80, 5)),
                          'min_samples_split': [2, 3, 4, 5, 6, 7],
                          'min_samples_leaf': [1, 2, 3, 4, 5],
                          'max_features': ['auto', 'sqrt', 'log2', None]}
        
    elif alg == 'dt':
        from sklearn.tree import DecisionTreeClassifier
        estimator = DecisionTreeClassifier()
        if param_vals == 'None':
            param_vals = {'max_depth': [2, 3, 5, 7, 9, 11, 13, 16],
                      'min_samples_split': [2, 3, 4, 5, 6, 7, 10],
                      'min_samples_leaf': [1, 2, 3, 4, 5],
                      'max_features': ['auto', 'sqrt', 'log2', None], 
                      'random_state': [0, 1, 5, 10, 50, 100]}
        
    elif alg == 'xgb':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(objective='binary:logistic', booster='gbtree', nthread=4, eval_metric='auc',
                                  use_label_encoder=False, random_state=est_rs)
        if param_vals == 'None':
            param_vals = {'max_depth': [2, 3, 4, 5, 7, 9, 11],
                      'n_estimators': list(np.arange(50, 200, 20)),
                      'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2],
                      'subsample': [0.5, 0.7, 0.8, 0.9, 1.0],
                      'reg_lambda': list(np.arange(0.1, 1.0, 0.1)),
                      'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 
                      'gamma': list(np.arange(1, 10, 1))}
        
    elif alg == 'gb':
        from sklearn.ensemble import GradientBoostingClassifier
        estimator = GradientBoostingClassifier()
        if param_vals == 'None':
            param_vals = {'max_depth': [2, 3, 5, 7, 9, 11, 13, 16],
                      'min_samples_split': [2, 3, 4, 5, 6, 7, 10],
                      'min_samples_leaf': [1, 2, 3, 4, 5],
                      'max_features': ['auto', 'sqrt', 'log2', None], 
                      'random_state': [0, 1, 5, 10, 50, 100]}
    elif alg == 'cat':
        from catboost import CatBoostClassifier
        cat_features = []
        estimator = CatBoostClassifier(loss_function='Logloss', nan_mode='Min', cat_features=cat_features,
                                       one_hot_max_size=31, random_state=est_rs)
        if param_vals == 'None':
            param_vals = {'max_depth': [4, 6, 8, 10, 12],
                      'n_estimators': list(np.arange(10, 100, 10)),
                      'learning_rate': [0.01, 0.05, 0.1, 0.2],
                      'subsample': [0.7, 0.8, 0.9, 1.0]}
        
    elif alg == 'sgd_elastic':
        from sklearn.linear_model import SGDClassifier # stochastic gradient descent (SGD)
        estimator = SGDClassifier(loss='log_loss', penalty='elasticnet')
        if param_vals == 'None':
            param_vals = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1], 
                      'l1_ratio': [0, 0.1, 0.15, 0.20, 0.25, 0.4, 0.5, 0.6, .75, 0.9, 1],
                      'max_iter': list(np.arange(10, 200, 20))}
        
    elif alg == 'sgd_l2':
        from sklearn.linear_model import SGDClassifier
        estimator = SGDClassifier(loss='log_loss', penalty='l2')
        if param_vals == 'None':
            param_vals = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1], 
                      'l1_ratio': [0, 0.1, 0.15, 0.20, 0.25, 0.4, 0.5, 0.6, .75, 0.9, 1],
                      'max_iter': list(np.arange(10, 200, 20))}
        
    elif alg == 'lr_l2': # Logistic Regression
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression(penalty='l2')
        if param_vals == 'None':
            param_vals = {'random_state': [0, 1, 5, 10,50,100], 
                      'l1_ratio': [0, 0.1, 0.15, 0.20, 0.25, 0.4, 0.5, 0.6, .75, 0.9, 1],
                      'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        
    elif alg == 'lr':
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression()
        if param_vals == 'None':
            param_vals = {}
        
    elif alg == 'svm':
        from sklearn import svm
        estimator = svm.SVC(probability=True)
        if param_vals == 'None':
            param_vals = {'C': [0.1, 1, 2, 3, 8, 10, 50, 100, 1000],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                      'degree': [2, 3, 4, 5, 6],
                      'gamma': ['scale', 'auto']}
        
    elif alg == 'knn': # k-nearest neighbors
        from sklearn.neighbors import KNeighborsClassifier
        estimator = KNeighborsClassifier()
        if param_vals == 'None':
            param_vals = {'n_neighbors': [3, 5, 7, 9, 11, 15, 20, 25, 30],
                      'weights': ['uniform', 'distance'],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'leaf_size': [10, 20, 30, 40, 50]} 
        
    elif alg == 'mlp': # Multi-layer Perceptron
        from sklearn.neural_network import MLPClassifier
        estimator = MLPClassifier(random_state=est_rs)
        if param_vals == 'None':
            param_vals = {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                      'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'solver': ['lbfgs', 'sgd', 'adam'],
                      'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
                      'learning_rate': ['constant', 'invscaling', 'adaptive']}
    
    elif alg == 'nb': # Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        estimator = GaussianNB()
        if param_vals == 'None':
            param_vals = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
    
    print("Parameter Values: ", param_vals)
    return estimator, param_vals
    

def setup_binary(set_up, experiment_name, project_name, unique_value_threshold=10):
    algorithm = set_up['algorithm']

    problem_type = set_up['problem_type']
    
    filename = set_up['filename']
    df = Common_Tools.upload_data(filename)
    
    input_cols, label_cols, categorical_cols, numeric_cols = Common_Tools.parse_exp_csv(df, filename, project_name, unique_value_threshold=unique_value_threshold)
    
    #print("Input Columns: ", input_cols)
    #print("Label Columns: ", label_cols)
    #print("Categorical Columns: ", categorical_cols)
    #print("Numeric Columns: ", numeric_cols)
    #print("Label values: ", df[label_cols].unique())
    
    #first_Feature = set_up['first_Feature']
    #last_Feature = set_up['last_Feature']
    #output_label = set_up['output_label']

    first_label = df[label_cols].unique()[0]
    second_label = df[label_cols].unique()[1]
    
    print("first_label: ", first_label)
    print("second_label: ", second_label)
    
    cross_validation = set_up['cross_validation']
    print("Cross Validation: ", cross_validation)
    
    options = set_up['options']
    
    param_vals = set_up['param_vals']
    
    # Write to text file
    algorithm_folder = os.path.join(project_name, experiment_name)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    with open(os.path.join(algorithm_folder, "parameter_setup.txt"), "w") as file:
        file.write('Experiment Name: %s\n' % experiment_name)
        file.write('Data Set Name: %s\n' % filename)
        file.write('ML Algorithm: %s\n' % algorithm)
        file.write('Problem Type: %s\n' % problem_type)
        file.write('Cross Validation? : %s\n' % cross_validation)
        for key, value in options.items(): 
            file.write('%s:%s\n' % (key, value))
    file.close()
    
    #input_cols = df.loc[:, first_Feature: last_Feature].columns.tolist()
    #label_cols = outputs_cols

    #input_df = df.loc[:, input_cols]

    #categorical_cols = input_df.select_dtypes('object').columns.tolist()
    #numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    
    df[label_cols] = df[label_cols].fillna(df[label_cols].mode()[0])

    if first_label in df[label_cols].tolist() and second_label in df[label_cols].tolist():
        df[label_cols] = df[label_cols].map({first_label: 0, second_label: 1}).astype(int)
    df[label_cols]
    
    return algorithm, df, algorithm_folder, input_cols, label_cols, categorical_cols, numeric_cols, cross_validation, options, param_vals


def setup_test_set_binary(set_up, experiment_name, project_name, unique_value_threshold=10):
    algorithm = set_up['algorithm']

    problem_type = set_up['problem_type']
    
    filename = set_up['filename']
    df = Common_Tools.upload_data(filename)
    
    test_set = set_up['test_set']
    df_test = Common_Tools.upload_data(test_set)
    
    input_cols, label_cols, categorical_cols, numeric_cols = Common_Tools.parse_exp_csv(df, filename, project_name, unique_value_threshold=unique_value_threshold)
    
    #print("Input Columns: ", input_cols)
    #print("Label Columns: ", label_cols)
    #print("Categorical Columns: ", categorical_cols)
    #print("Numeric Columns: ", numeric_cols)
    #print("Label values: ", df[label_cols].unique())
    
    #first_Feature = set_up['first_Feature']
    #last_Feature = set_up['last_Feature']
    #output_label = set_up['output_label']

    first_label = df[label_cols].unique()[0]
    second_label = df[label_cols].unique()[1]
    
    print("first_label: ", first_label)
    print("second_label: ", second_label)
    
    options = set_up['options']
    
    param_vals = set_up['param_vals']
    
    # Write to text file
    algorithm_folder = os.path.join(project_name, experiment_name)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    with open(os.path.join(algorithm_folder, "parameter_setup.txt"), "w") as file:
        file.write('Experiment Name: %s\n' % experiment_name)
        file.write('Data Set Name (Train): %s\n' % filename)
        file.write('Data Set Name (Test): %s\n' % test_set)
        file.write('ML Algorithm: %s\n' % algorithm)
        file.write('Problem Type: %s\n' % problem_type)
        for key, value in options.items(): 
            file.write('%s:%s\n' % (key, value))
    file.close()
    
    #input_cols = df.loc[:, first_Feature: last_Feature].columns.tolist()
    #label_cols = outputs_cols

    #input_df = df.loc[:, input_cols]

    #categorical_cols = input_df.select_dtypes('object').columns.tolist()
    #numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    
    df[label_cols] = df[label_cols].fillna(df[label_cols].mode()[0])

    if first_label in df[label_cols].tolist() and second_label in df[label_cols].tolist():
        df[label_cols] = df[label_cols].map({first_label: 0, second_label: 1}).astype(int)
    df[label_cols]
    
    df_test[label_cols] = df_test[label_cols].fillna(df_test[label_cols].mode()[0])

    if first_label in df_test[label_cols].tolist() and second_label in df_test[label_cols].tolist():
        df_test[label_cols] = df_test[label_cols].map({first_label: 0, second_label: 1}).astype(int)
    df_test[label_cols]
    
    return algorithm, df, df_test, algorithm_folder, input_cols, label_cols, categorical_cols, numeric_cols, options, param_vals



# get the correlation matrix
def correlation_matrix(df, numeric_cols, label_cols, algorithm_folder):
    # Compute the correlation matrix
    ness_cols = numeric_cols.copy()
    ness_cols.append(label_cols)
    correlation_matrix = df[ness_cols].corr()

    # Set up the figure and axes
    plt.figure(figsize=(10, 8))

    # Create the heatmap using Seaborn
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    
    # Adjust font size dynamically based on the size of the heatmap cells
    n_rows, n_cols = correlation_matrix.shape
    cell_size = 1.0 / max(n_rows, n_cols)
    font_size = 150 * cell_size  # Adjust multiplier as needed

    # Annotate values inside the matrix with count and percentage
    for i in range(len(correlation_matrix)):
        for j in range(len(correlation_matrix)):
            plt.text(j + 0.5, i + 0.5, f'{correlation_matrix.iloc[i, j]:.2f}', horizontalalignment='center', verticalalignment='center', color='black', fontsize=font_size)

    # Set plot title
    plt.title('Correlation Heatmap')
    
    # Save the figure
    algorithm_folder = algorithm_folder
    # Save the combined figure as a .png file
    save_path = os.path.join(algorithm_folder, 'correlation_matrix.png')
    plt.savefig(save_path)
    plt.show()

# get the top 20 varaibles correlated with the final output label to predict
def top_correlation_variables(df, numeric_cols, label_cols, algorithm_folder):
    # Compute the correlation matrix
    ness_cols = numeric_cols.copy()
    ness_cols.append(label_cols)
    correlation_matrix = df[ness_cols].corr()
    
    # Calculate the correlation of 'A' with other columns
    corr_with_label = df[numeric_cols].corrwith(df[label_cols])
    
    # Get the top 20 variables corr. with the output label
    top_corr_variables = corr_with_label.abs().sort_values(ascending=False)[:21]

    # Print the top 20 variables corr. with column label
    print("Top 20 Correlated Variables with " + label_cols + " (By absolute values) :")
    print(top_corr_variables)
    
    # Write to text file
    algorithm_folder = algorithm_folder
    with open(os.path.join(algorithm_folder, "top_correlated_variables.txt"), "w") as file:
        file.write("Top 20 Correlated Variables with '{}' (By absolute values) :\n".format(label_cols))
        for variable, correlation in top_corr_variables.items():
            file.write("{} : {}\n".format(variable, correlation))


def plot_and_save_confusion_matrix_rate(avg_conf_matrix, algorithm, classes, algorithm_folder):
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_conf_matrix, cmap=plt.cm.Blues)
    plt.title(f'Average Confusion Matrix - {algorithm}')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
        
    # Annotate values inside the matrix
    for i in range(len(avg_conf_matrix)):
        for j in range(len(avg_conf_matrix[i])):
            plt.text(j, i, f'{avg_conf_matrix[i][j]:.2f}', horizontalalignment='center', verticalalignment='center', color='black')
                
    # Set ticks and labels for x and y axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
        
    # Save the combined figure as a .png file
    save_path = os.path.join(algorithm_folder, f'{algorithm}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()  # Close the plot to avoid displaying in console


# plot one confusion matrix
def plot_and_save_confusion_matrix_percentage(avg_conf_matrix, algorithm, classes, algorithm_folder):   
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_conf_matrix, cmap=plt.cm.Reds)
    plt.title(f'Average Confusion Matrix - {algorithm}')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
        
    # Annotate values inside the matrix with count and percentage
    for i in range(len(avg_conf_matrix)):
        for j in range(len(avg_conf_matrix[i])):
            count = avg_conf_matrix[i][j]
            percentage = (count / np.sum(avg_conf_matrix)) * 100
            plt.text(j, i, f'{count:.0f} ({percentage:.2f}%)', horizontalalignment='center', verticalalignment='center', color='black', fontsize=10)
                
    # Set ticks and labels for x and y axes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
        
    # Save the combined figure as a .png file
    save_path = os.path.join(algorithm_folder, f'{algorithm}_confusion_matrix.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()  # Close the plot to avoid displaying in console
        
def plot_auc_scores_fold(algorithm, roc_auc_list_test, mean_auc_test, algorithm_folder):
    """
    Plot the AUC score across each fold for the given algorithm for the test set.

    Parameters:
        algorithm (str): The name of the algorithm.
        roc_auc_list_test (list): List of AUC scores for each fold.
        mean_auc_test (float): Mean AUC score for the algorithm.
        algorithm_folder (str): Path to the folder where the plot will be saved.
    """
    plt.figure()
    plt.plot(roc_auc_list_test, label="AUC Score:=" + str(mean_auc_test))
    plt.title('Test Set AUC Scores for ' + algorithm)
    plt.xlabel('Fold')
    plt.ylabel('AUC Score')
    plt.legend(loc=0)
    plt.savefig(os.path.join(algorithm_folder, algorithm + "_aucscores.png"))
    plt.close()


def plot_mean_roc_curve(algorithm, mean_fpr, mean_tpr_train, mean_tpr_test, mean_auc_train, mean_auc_test, tprs_lower_train, tprs_upper_train, tprs_lower_test, tprs_upper_test, conf_int_train, conf_int_test, algorithm_folder, youden="False"):
    """
    Plot the average ROC/AUC curve for the given algorithm for cross validation.

    Parameters:
        algorithm (str): The name of the algorithm.
        mean_fpr (array-like): Mean false positive rates.
        mean_tpr_train (array-like): Mean true positive rates for the training set.
        mean_tpr_test (array-like): Mean true positive rates for the test set.
        mean_auc_train (float): Mean AUC score for the training set.
        mean_auc_test (float): Mean AUC score for the test set.
        tprs_lower_train (array-like): Lower bound true positive rates for the training set.
        tprs_upper_train (array-like): Upper bound true positive rates for the training set.
        tprs_lower_test (array-like): Lower bound true positive rates for the test set.
        tprs_upper_test (array-like): Upper bound true positive rates for the test set.
        algorithm_folder (str): Path to the folder where the plot will be saved.
    """
    if youden == "True":
        # Calculate the Youden index
        youden_index = np.argmax(mean_tpr_test - mean_fpr)

    fig_auc = plt.figure(figsize=(8, 6))
    plt.plot(mean_fpr, mean_tpr_train, label="Average ROC/AUC Curve (Train Set), AUC={:.3f}, CI: [{:.3f}-{:.3f}]".format(mean_auc_train, conf_int_train[0], conf_int_train[1]))
    plt.plot(mean_fpr, mean_tpr_test, label="Average ROC/AUC Curve (Test Set), AUC={:.3f}, CI: [{:.3f}-{:.3f}]".format(mean_auc_test, conf_int_test[0], conf_int_test[1]))
    plt.plot([0, 1], [0, 1], color='g')
    if youden == "True":
        plt.scatter(mean_fpr[youden_index], mean_tpr_test[youden_index], c='red', marker='o', label='Youden Index')
    plt.title('Average AUC for ' + algorithm)
    plt.legend(loc=0)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.fill_between(mean_fpr, tprs_lower_test, tprs_upper_test, color='orange', alpha=.15)
    plt.fill_between(mean_fpr, tprs_lower_train, tprs_upper_train, color='green', alpha=.15)
    plt.savefig(os.path.join(algorithm_folder, algorithm + "_auc.png"))
    plt.show()
    
    return mean_fpr, mean_tpr_test, mean_auc_test, conf_int_test[0], conf_int_test[1]


def plot_test_accuracy(algorithm, test_accuracy_list, test_acc, test_dummy_list, test_dummy, algorithm_folder):
    """
    Plot the average test accuracy score for the given algorithm.

    Parameters:
        algorithm (str): The name of the algorithm.
        test_accuracy_list (list): List of test accuracy scores for each fold.
        test_acc (float): Average test accuracy score.
        test_dummy_list (list): List of dummy accuracy scores for each fold.
        test_dummy (float): Average dummy accuracy score.
        algorithm_folder (str): Path to the folder where the plot will be saved.
    """
    plt.figure()
    plt.plot(test_accuracy_list, label="Test Accuracy Score. Avg:=" + str(test_acc))
    plt.plot(test_dummy_list, label="Dummy Accuracy Score. Avg:=" + str(test_dummy))
    plt.title('Test and Dummy Accuracy for ' + algorithm)
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)
    plt.savefig(os.path.join(algorithm_folder, algorithm + "_test_accuracy.png"))
    plt.show()
    plt.close()

def plot_shap(shap_values_list, X_test_fold, output_df, algorithm, algorithm_folder):
    # Compute and plot averaged SHAP values across folds
    plt.figure()
    mean_shap_values = np.mean(shap_values_list, axis=0)
    print(X_test_fold.shape)
    print(mean_shap_values.shape)
    shap.summary_plot(mean_shap_values, X_test_fold, plot_type='dot', max_display = 10, show=False) 
    filename_shap = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_shap.png")
    plt.savefig(filename_shap,dpi=700)
    plt.show()  # Display the plot

def save_performance_metrics(algorithm_folder, algorithm, mean_tpr, mean_fpr_list, mean_tnr, mean_fnr, mean_ppv, mean_npv, f1_mean, dor_mean, z_score_test, p_value_test, mean_auc_test, test_acc):
    # Create an empty DataFrame to store performance metrics
    res_array = pd.DataFrame(columns=['tpr', 'fpr', 'tnr', 'fnr', 'ppv', 'npv', 'f1', 'dor', 'z-score:', 'p-value', 'AUC', 'Test Set Accuracy'])
    
    # Assign values to DataFrame columns
    res_array.loc[0] = [mean_tpr, mean_fpr_list, mean_tnr, mean_fnr, mean_ppv, mean_npv, f1_mean, dor_mean, z_score_test, p_value_test, mean_auc_test, test_acc]

    # Print the performance metrics DataFrame
    print(res_array)

    # Save the DataFrame to a CSV file
    filename = os.path.join(algorithm_folder, algorithm + "_stats_table.csv")
    res_array.to_csv(filename, index=False)

def plot_precision_recall_curve(precision_test_list, mean_recall, p_r_auc_list, algorithm, algorithm_folder):
    # Compute the average precision and recall and its AUC from all folds
    mean_precision = Common_Tools.average_values_across_lists(precision_test_list)
    #mean_recall = average_values_across_lists(recall_test_list)
    mean_p_r_auc = np.mean(p_r_auc_list)
        
    # plot the precision-recall curves
    plt.plot([0, 1], [1, 0], linestyle='--')
    plt.plot(mean_recall, mean_precision, marker='.', label='Average AUC for Precision/Recall Curve:=' + str(mean_p_r_auc))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show title
    plt.title('Average Precision/Recall Curve for ' + algorithm)
    # show the legend
    plt.legend()
    # Save the combined figure as a .png file
    save_path = os.path.join(algorithm_folder, f'{algorithm}_precision_recall_curve.png')
    plt.savefig(save_path)
    # show the plot
    plt.show()
    
    return mean_p_r_auc

def bootstrap(y_train, y_predicted_train, y_test, y_predicted_test, n_bootstraps=1000):
    
    # Arrays to store bootstrap AUC values
    auc_train_bootstrap = np.zeros(n_bootstraps)
    auc_test_bootstrap = np.zeros(n_bootstraps)

    # Bootstrap resampling and AUC calculation
    for i in range(n_bootstraps):
        # Bootstrap resampling for training set
        indices_train = resample(np.arange(len(y_predicted_train)), replace=True)
        y_train_bootstrap = y_train.iloc[indices_train].values  # Get values corresponding to indices
        y_predicted_train_bootstrap = y_predicted_train[indices_train]
        fpr_train_bootstrap, tpr_train_bootstrap, _ = roc_curve(y_train_bootstrap, y_predicted_train_bootstrap)
        auc_train_bootstrap[i] = auc(fpr_train_bootstrap, tpr_train_bootstrap)

        # Bootstrap resampling for testing set
        indices_test = resample(np.arange(len(y_predicted_test)), replace=True)
        y_test_bootstrap = y_test.iloc[indices_test].values  # Get values corresponding to indices
        y_predicted_test_bootstrap = y_predicted_test[indices_test]
        fpr_test_bootstrap, tpr_test_bootstrap, _ = roc_curve(y_test_bootstrap, y_predicted_test_bootstrap)
        auc_test_bootstrap[i] = auc(fpr_test_bootstrap, tpr_test_bootstrap)

    # Calculate confidence intervals
    conf_int_train = np.percentile(auc_train_bootstrap, [2.5, 97.5])
    conf_int_test = np.percentile(auc_test_bootstrap, [2.5, 97.5])
    
    return conf_int_train, conf_int_test

def roc_auc_curve(y_train, probas_train, y_test, probas_test, algorithm, algorithm_folder, youden="False", youden_index="None"):
    # Calculate ROC/AUC scores on train set
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, probas_train[:, 1])
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    
    # Calculate ROC/AUC scores on test set
    fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, probas_test[:, 1])
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
                
    # Compute the bootstrap confidence intervals for ROC Curves
    conf_int_train, conf_int_test = bootstrap(y_train, probas_train[:, 1], y_test, probas_test[:, 1])
    
    if youden == "True":
        # Calculate the Youden index
        youden_cutoff = np.argmax(tpr_test - fpr_test)
        
    print("Youden Index: ", youden_index)
                               
    fig_auc = plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_train, tpr_train, label='(AUC Training Score = {:.3f}, CI: {:.3f}-{:.3f})'.format(roc_auc_train, conf_int_train[0], conf_int_train[1]))
    plt.plot(fpr_test, tpr_test, label='(AUC Test Score = {:.3f}, CI: {:.3f}-{:.3f})'.format(roc_auc_test, conf_int_test[0], conf_int_test[1]))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Average AUC for ' + algorithm)
    if youden == "True":
        plt.scatter(fpr_test[youden_cutoff], tpr_test[youden_cutoff], c='red', marker='o', label='Youden Index')
    plt.legend()
    filename_roc = os.path.join(algorithm_folder, algorithm + "_auc.png")
    plt.savefig(filename_roc,dpi=700)
    plt.show()    
    
    return fpr_test, tpr_test, roc_auc_test, conf_int_test[0], conf_int_test[1]
            
def p_r_curve(y_test, probas_test, algorithm, algorithm_folder):
    # calcuate the percision and recall on test set and the AUC for it
    precision_test, recall_test, _ = precision_recall_curve(y_test, probas_test[:, 1])
    p_r_auc = auc(recall_test, precision_test)
                               
    # plot the precision-recall curves
    plt.plot([0, 1], [1, 0], linestyle='--')
    plt.plot(recall_test, precision_test, marker='.', label='AUC for Precision/Recall Curve:=' + str(p_r_auc))
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show title
    plt.title('Precision/Recall Curve for ' + algorithm)
    # show the legend
    plt.legend()
    # Save the combined figure as a .png file
    save_path = os.path.join(algorithm_folder, f'{algorithm}_precision_recall_curve.png')
    plt.savefig(save_path)
    # show the plot
    plt.show()

def save_metrics_singluar(tpr, fpr, tnr, fnr, ppv, npv, f1, dor, roc_auc_test, test_acc, cutoff, algorithm, algorithm_folder):
    # get full performance metrics
    # return full performance metrics, and youden index based
    res_array = pd.DataFrame(columns=['tpr', 'fpr', 'tnr', 'fnr', 'ppv', 'npv', 'f1', 'dor', 'AUC','Test Set Accuracy','Youdens index cuttoff'])
    # Assign values to DataFrame columns
    res_array.loc[0] = [tpr, fpr, tnr, fnr, ppv, npv, f1, dor, roc_auc_test, test_acc, cutoff]
    
    # save table to csv file
    filename = os.path.join(algorithm_folder, algorithm + "_stats_table.csv")
    res_array.to_csv(filename, index=False)
        
    print(res_array)

def train_test_split_binary(df, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals_raw, experiment_name, project_name):
    '''
    This function is for the tradtional train test split process where you have one data set and you split it into a training and test set.
    This function does NOT do any cross validations.
    This could be a MODEL CREATION function.
    '''
    #Seperate the inputs and outputs for test data
    input_df, output_df = Common_Tools.split(df, input_cols, label_cols)
    
    print(options['test_size'])
    
    # Split the data between train and test
    X_train, X_test, y_train, y_test = train_test_split(input_df, output_df, test_size=options['test_size'], stratify=output_df,random_state=42)
    
    # Before the cross-validation loop
    classes = output_df.unique()
    
    print("Classes: ", classes)
    
    algorithm_folder = os.path.join(project_name, experiment_name, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    filename = os.path.join(algorithm_folder, algorithm + ".txt")
    f = open(filename, "w")
    print("Algorithm:", algorithm)
    print("Label:", output_df.name)
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algorithm)
    f.write("\nLabel: %s"% output_df.name)
        
    # get the specified model/estimator
    estimator, param_vals = get_classifier(algorithm, param_vals_raw)
            
    X_col = X_train.columns.to_list()
            
    # Imputing Data if chosen
    if options['Impute'] == "True":
        print("Impute")
        imputer = SimpleImputer(strategy = 'mean')
        imputer.fit(X_train)
        X_train = pd.DataFrame(imputer.transform(X_train), columns = X_col)
        #df[numeric_cols] = imputer.transform(df[numeric_cols])
        X_test = pd.DataFrame(imputer.transform(X_test), columns = X_col)
        y_train.reset_index(drop=True, inplace=True)
                
    # Scaling if chosen
    if options['Scaling'] == "True":
        X_train, scaler = Common_Tools.scaling(X_train, input_cols, label_cols, numeric_cols, categorical_cols, scalingMethod=options['scalingMethod'])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        y_train.reset_index(drop=True, inplace=True)
                
    # Rebalance Data if chosen
    if options['rebalance'] == "True":
        print('rebalance')
        X_train, y_train = Common_Tools.rebalance(X_train, y_train, options['rebalance_type'])
    # Feature Selection if chosen
    if options['FeatureSelection'] == "True":
        X_train, selected_features = Common_Tools.feature_selection(X_train, y_train, method=options['method'], type=options['type'], N=options['N_features'], per=options['per'])
        #print(selected_features)
        X_test = X_test[selected_features]
        f.write("\nSelected Features: %s"%selected_features)
    else:
        selected_features = list(X_train.columns)
                
    estimator = Common_Tools.train_tune(estimator, param_vals, X_train, y_train, options['strategy'], itr=options['itr'])
            
    # Calibrate the model if chosen
    if options['calibrate'] == "True":
        estimator = Common_Tools.calibrate(estimator, X_train, y_train)
            
    params = estimator.get_params()
    f.write("\nParameters: %s"%params)
            
    predictions_train = estimator.predict(X_train) # predict with test set
    probas_train = estimator.predict_proba(X_train)
    predictions_test = estimator.predict(X_test) # predict with test set
    probas_test = estimator.predict_proba(X_test) # get probablities with test set
            
    # Calibrate the model if chosen
    if options['calibrate'] == "True":
        prob_true, prob_pred = calibration_curve(y_test, probas_test[:, 1], n_bins=10)
        disp = CalibrationDisplay(prob_true, prob_pred, probas_test[:, 1])
        print(disp)
        disp.plot()
                
        # Append these probabilities to the lists
        print("Prob Sizes: ")
        print(prob_true.shape)
        print(prob_pred.shape)
            
    train_acc = accuracy_score(y_train, predictions_train) # training accuracy
    test_acc = accuracy_score(y_test, predictions_test) # test accuracy

    if options['dummy_strategy'] == "all zeros":
        dumb_acc = accuracy_score(y_test, Common_Tools.all_zero(X_test, y_test)) # use a dummy model  where it predicts all 0's to comapre to trained model
    elif options['dummy_strategy'] == "all ones":
        dumb_acc = accuracy_score(y_test, Common_Tools.all_ones(X_test, y_test)) # use a dummy model  where it predicts all 1's to comapre to trained model
               
    f.write("\nTraining Accuracy: %.2f%%" % (train_acc * 100))
    print("Training Accuracy:", (train_acc * 100))
    f.write("\nTest Accuracy: %.2f%%" % (test_acc * 100))
    print("Test Accuracy:", (test_acc * 100))
    f.write("\nDummy Model (Predict all 0) Accuracy vs. Test: %.2f%%" % (dumb_acc * 100))
    print("Dummy Model (Predict all 0) Accuracy vs. Test:", (dumb_acc * 100))

    # Calculate confusion matrix
    if options['confusion_matrix'] == 'Rate':
        conf_matrix = confusion_matrix(y_test, predictions_test, normalize='true') # normalize to get tpr, fpr, etc.
        plot_and_save_confusion_matrix_rate(conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on rates (TPR, FPR, etc.)
    elif options['confusion_matrix'] == 'Percentage':
        conf_matrix = confusion_matrix(y_test, predictions_test) # gets pure number of tp's, fp's, etc.
        plot_and_save_confusion_matrix_percentage(conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on the number of TP's, FP's, etc. and thier percentage    
            
    # calculate P and N
    P = np.sum(list(y_test))
    N = len(y_test) - P
            
    # Calculating true negatives, false psotives, false negatives, and true postives
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
            
    # Calculating TPR (True Positive Rate) and FPR (False Positive Rate) as induivuial values
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
            
    # Calculating TNR (True Negative Rate) and FNR (False Negative Rate) as induivuial values
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
            
    ppv = (tpr * P) / ((tpr * P) + (fpr * N))
    npv = (tnr * N) / ((tnr * N) + (fnr * P))
    f1 = (2.0 * ppv * tpr) / (ppv + tpr)
    dor = (tpr * tnr) / (fpr * fnr)
    
    youden_index = (tp / (tp + fn)) + (tn / (fp + tn)) - 1
                
    fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = roc_auc_curve(y_train, probas_train, y_test, probas_test, algorithm, algorithm_folder, options['Youden'],  youden_index)                           
            
    p_r_curve(y_test, probas_test, algorithm, algorithm_folder)

    filename_prob = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_prob.joblib")
    filename_pred = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_pred.joblib")
    filename_true = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_true_fullset.joblib")
    filename_features = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_selected_features.joblib")

    joblib.dump(probas_test, filename_prob) # test set probablities
    joblib.dump(predictions_test, filename_pred) # test set predictions
    joblib.dump(output_df, filename_true, compress=1) # full set acutal labels
    joblib.dump(selected_features, filename_features, compress=1) # selected features
            
    modelname = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_model.joblib")
    joblib.dump(estimator, modelname)
            
    #""" SHAP Values
    if options['SHAP'] == "True":
        #explaining model
        print(X_test.shape)
        explainer = shap.Explainer(estimator.predict, X_test)
        shap_values = explainer(X_test, max_evals=2000)
        print(shap_values.shape)
                
        shap_values_array = shap_values.values  # Extract SHAP values from Explanation objects
                
        shap.summary_plot(shap_values_array, X_test, max_display=10, show=False)
        
        filename_shap = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_shap.png")
        plt.savefig(filename_shap,dpi=700)
        plt.show()
                               
        plt.close()
                               
    f.write('\n')
        
    # get full performance metrics
    save_metrics_singluar(tpr, fpr, tnr, fnr, ppv, npv, f1, dor, roc_auc_test, test_acc, youden_index, algorithm, algorithm_folder)

    
    f.close()
    
    return fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u

def cross_validation_stratified_binary(df, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals_raw, experiment_name, project_name):
    '''
    This function is for the stratified cross validation function.
    This is for model evaluation, NOT model creation.
    '''

    #Seperate the inputs and outputs for test data
    input_df, output_df = Common_Tools.split(df, input_cols, label_cols)
    
    # Before the cross-validation loop
    classes = output_df.unique()
    
    print("Classes: ", classes)
    
    algorithm_folder = os.path.join(project_name, experiment_name, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    filename = os.path.join(algorithm_folder, algorithm + ".txt")
    f = open(filename, "w")
    print("Algorithm:", algorithm)
    print("Label:", output_df.name)
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algorithm)
    f.write("\nLabel: %s"% output_df.name)
        
    # Lists to store ROC curve information for each fold
    tpr_list_train = [] # train set
    mean_fpr = np.linspace(1, 0, 100)
    roc_auc_list_train = [] # list of ROC/AUC Scores for each fold
        
    tpr_list_test = [] # tpr for test set
    roc_auc_list_test = [] # list of ROC/AUC Scores for each fold
        
    tpr_list = [] # tpr list
    tnr_list = [] # tnr list
    fnr_list = [] # fnr list
    fpr_list = [] # fpr list
    ppv_list = [] # ppv list
    npv_list = [] # npv for test set
    f1_list = [] # f1 for test set
    dor_list = [] # dor for test set
        
    test_accurcy_list = [] # test accuracy scores
    test_dummy_list = [] # test dummy scores
        
    # Initialize a list to store SHAP values for each fold
    shap_values_list = []
        
    # Initialize lists to store true and predicted probabilities across all folds
    all_prob_true = []
    all_prob_pred = []
        
    # Lists to store P-R curve information for each fold
    precision_test_list = [] # precision test
    recall_test_list = [] # recall test
    p_r_auc_list = [] # auc for precsion and recal curve
    mean_recall = np.linspace(0, 1, 100)
        
    # Inside the cross-validation loop
    sum_conf_matrix = None
        
    # Loop 10 times in StratifiedKFold
    for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=options['CV'], shuffle=True, random_state=42).split(input_df, output_df)):
            
        # get the specified model/estimator
        estimator, param_vals = get_classifier(algorithm, param_vals_raw)
            
        print(fold)
        f.write("\nFold: %s"%fold)
            
        #split training data into train and test sets
        X_train_fold, X_test_fold = input_df.iloc[train_idx], input_df.iloc[test_idx]
        y_train_fold, y_test_fold = output_df.iloc[train_idx], output_df.iloc[test_idx]
            
        X_col = X_train_fold.columns.to_list()
            
        # Imputing Data if chosen
        if options['Impute'] == "True":
            print("Impute")
            imputer = SimpleImputer(strategy = 'mean')
            imputer.fit(X_train_fold)
            X_train_fold = pd.DataFrame(imputer.transform(X_train_fold), columns = X_col)
            #df[numeric_cols] = imputer.transform(df[numeric_cols])
            X_test_fold = pd.DataFrame(imputer.transform(X_test_fold), columns = X_col)
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Scaling if chosen
        if options['Scaling'] == "True":
            X_train_fold, scaler = Common_Tools.scaling(X_train_fold, input_cols, label_cols, numeric_cols, categorical_cols, scalingMethod=options['scalingMethod'])
            X_test_fold[numeric_cols] = scaler.transform(X_test_fold[numeric_cols])
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Rebalance Data if chosen
        if options['rebalance'] == "True":
            print('rebalance')
            X_train_fold, y_train_fold = Common_Tools.rebalance(X_train_fold, y_train_fold, options['rebalance_type'])
        # Feature Selection if chosen
        if options['FeatureSelection'] == "True":
            X_train_fold, selected_features = Common_Tools.feature_selection(X_train_fold, y_train_fold, method=options['method'], type=options['type'], N=options['N_features'], per=options['per'])
            #print(selected_features)
            X_test_fold = X_test_fold[selected_features]
            f.write("\nSelected Features: %s"%selected_features)
        else:
            selected_features = list(X_train_fold.columns)
                
        estimator = Common_Tools.train_tune(estimator, param_vals, X_train_fold, y_train_fold, options['strategy'], itr=options['itr'])
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            estimator = Common_Tools.calibrate(estimator, X_train_fold, y_train_fold)
            
        params = estimator.get_params()
        f.write("\nParameters: %s"%params)
            
        predictions_train = estimator.predict(X_train_fold) # predict with test set
        probas_train = estimator.predict_proba(X_train_fold)
        predictions_test = estimator.predict(X_test_fold) # predict with test set
        probas_test = estimator.predict_proba(X_test_fold) # get probablities with test set
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            prob_true, prob_pred = calibration_curve(y_test_fold, probas_test[:, 1], n_bins=10)
            disp = CalibrationDisplay(prob_true, prob_pred, probas_test[:, 1])
            print(disp)
            disp.plot()
                
            # Append these probabilities to the lists
            print("Prob Sizes: ")
            print(prob_true.shape)
            print(prob_pred.shape)
            all_prob_true.append(prob_true)
            all_prob_pred.append(prob_pred)
            
        train_acc = accuracy_score(y_train_fold, predictions_train) # training accuracy
        test_acc = accuracy_score(y_test_fold, predictions_test) # test accuracy
        test_accurcy_list.append(test_acc)
        if options['dummy_strategy'] == "all zeros":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_zero(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 0's to comapre to trained model
        elif options['dummy_strategy'] == "all ones":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_ones(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 1's to comapre to trained model
        test_dummy_list.append(dumb_acc)
               
        f.write("\nTraining Accuracy: %.2f%%" % (train_acc * 100))
        print("Training Accuracy:", (train_acc * 100))
        f.write("\nTest Accuracy: %.2f%%" % (test_acc * 100))
        print("Test Accuracy:", (test_acc * 100))
        f.write("\nDummy Model (Predict all 0) Accuracy vs. Test: %.2f%%" % (dumb_acc * 100))
        print("Dummy Model (Predict all 0) Accuracy vs. Test:", (dumb_acc * 100))

        # Calculate confusion matrix for this fold
        if options['confusion_matrix'] == 'Rate':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test, normalize='true') # normalize to get tpr, fpr, etc.
        elif options['confusion_matrix'] == 'Percentage':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test) # gets pure number of tp's, fp's, etc.
            
            
        # calculate P and N
        P = np.sum(list(y_test_fold))
        N = len(y_test_fold) - P
            
        # Calculating true negatives, false psotives, false negatives, and true postives
        tn, fp, fn, tp = confusion_matrix(y_test_fold, predictions_test).ravel()
            
        # Calculating TPR (True Positive Rate) and FPR (False Positive Rate) as induivuial values
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
            
        # Calculating TNR (True Negative Rate) and FNR (False Negative Rate) as induivuial values
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
            
        ppv = (tpr * P) / ((tpr * P) + (fpr * N))
        npv = (tnr * N) / ((tnr * N) + (fnr * P))
        f1 = (2.0 * ppv * tpr) / (ppv + tpr)
        dor = (tpr * tnr) / (fpr * fnr)
            
        tpr_list.append(tpr)
        fnr_list.append(fnr)
        tnr_list.append(tnr)
        fpr_list.append(fpr)
        ppv_list.append(ppv)
        npv_list.append(npv)
        f1_list.append(f1)
        dor_list.append(dor)
            
            
        # summed up the confusion matrix
        if options['confusion_matrix'] != "None":
            if sum_conf_matrix is None:
                sum_conf_matrix = conf_matrix
            else:
                sum_conf_matrix += conf_matrix
                
        # Calculate ROC/AUC scores for each fold on train set
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train_fold, probas_train[:, 1])
        roc_auc_train = metrics.auc(fpr_train, tpr_train)
        roc_auc_list_train.append(roc_auc_train)
            
        # Calculate ROC/AUC scores for each fold on test set
        fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test_fold, probas_test[:, 1])
        roc_auc_test = metrics.auc(fpr_test, tpr_test)
        roc_auc_list_test.append(roc_auc_test)
                
        # Interpolate the ROC curve at a common set of points
        tpr_list_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tpr_list_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            
        # calculate fnr_test and tnr_test
        fnr_test = 1.0 - tpr_test
        tnr_test = 1.0 - fpr_test
            
        # calcuate the percision and recall on test set and the AUC for it
        precision_test, recall_test, _ = precision_recall_curve(y_test_fold, probas_test[:, 1])
        p_r_auc = auc(recall_test, precision_test)
        # Add to list
        precision_test_list.append(np.interp(mean_recall, precision_test, recall_test))
        p_r_auc_list.append(p_r_auc)

        filename_prob = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_prob.joblib")
        filename_pred = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_pred.joblib")
        filename_true = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_true_fullset.joblib")
        filename_features = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_selected_features.joblib")

        joblib.dump(probas_test, filename_prob) # test set probablities
        joblib.dump(predictions_test, filename_pred) # test set predictions
        joblib.dump(output_df, filename_true, compress=1) # full set acutal labels
        joblib.dump(selected_features, filename_features, compress=1) # selected features
            
        modelname = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_model.joblib")
        joblib.dump(estimator, modelname)
            
        #""" SHAP Values
        if options['SHAP'] == "True":
            #explaining model
            print(X_test_fold.shape)
            explainer = shap.Explainer(estimator.predict, X_test_fold)
            shap_values_fold = explainer(X_test_fold, max_evals=2000)
            print(shap_values_fold.shape)
                
            shap_values_fold_array = shap_values_fold.values  # Extract SHAP values from Explanation objects
                
            #shap.summary_plot(shap_values_fold_array, X_train_fold, max_display=10, show=True)     
                
            # Append SHAP values to the list
            shap_values_list.append(shap_values_fold_array)
            print("List Shape:",np.shape(shap_values_list))
                
                
        #"""
        f.write('\n')
        
    # Compute average tpr scores across all folds
    mean_tpr_test = np.mean(tpr_list_test, axis=0)
    std_tpr_test = np.std(tpr_list_test, axis=0)
    mean_tpr_train = np.mean(tpr_list_train, axis=0)
    std_tpr_train = np.std(tpr_list_train, axis=0)

    # Compute average ROC/AUC scores across all folds for train set
    #mean_auc_train = np.mean(roc_auc_list_train) # train set
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    std_auc_train = np.nanstd(roc_auc_list_train) # std
        
    # get confidence interval for test AUC score for train set
    se_tpr_train = std_tpr_train/np.sqrt(options['CV'])
    se_auc_train = std_auc_train/np.sqrt(options['CV'])
    t_score_train = stats.t.ppf(1-0.025, options['CV']) # t-score
    tprs_upper_train = np.minimum(mean_tpr_train + t_score_train*se_tpr_train, 1)
    tprs_lower_train = np.maximum(mean_tpr_train - t_score_train*se_tpr_train, 0)
    cilow = np.maximum(mean_auc_train-t_score_train*se_auc_train, 0) # lower bound
    cihigh = np.minimum(mean_auc_train+t_score_train*se_auc_train, 1) # upper bound
    conf_int_train = [cilow, cihigh]
        
    f.write("\nAverage AUC Score (Train Set): %s" % mean_auc_train)
    print("Average AUC Score (Train Set):", mean_auc_train)
    f.write("\n95%% Confidence Interval (Train Set): [%0.2f, %0.2f]" % (conf_int_train[0], conf_int_train[1]))
    print("95% Confidence Interval (Train Set):", conf_int_train)

    # Compute average ROC/AUC scores across all folds for test set
    mean_auc_test = auc(mean_fpr, mean_tpr_test) # test set
    std_auc_test = np.nanstd(roc_auc_list_test) # std
    
    # get confidence interval for test AUC score for test set
    se_tpr_test = std_tpr_test/np.sqrt(options['CV'])
    se_auc_test = std_auc_test/np.sqrt(options['CV'])
    t_score_test = stats.t.ppf(1-0.025, options['CV'])
    tprs_upper_test = np.minimum(mean_tpr_test + t_score_test*se_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - t_score_test*se_tpr_test, 0) 
    cilow = np.maximum(mean_auc_test-t_score_test*se_auc_test, 0) # lower bound
    cihigh = np.minimum(mean_auc_test+t_score_test*se_auc_test, 1) # upper bound
    conf_int_test = [cilow, cihigh]
        
    f.write("\nAverage AUROC Score (Test Set): %s" % mean_auc_test)
    print("Average AUROC Score (Test Set):", mean_auc_test)
    f.write("\n95%% Confidence Interval (Test Set): [%0.2f, %0.2f]" % (conf_int_test[0], conf_int_test[1]))
    print("95% Confidence Interval (Test Set):", conf_int_test)

    auc_null = 0.5  # Null hypothesis: AUC = 0.5
    z_score_test = mean_auc_test / np.sqrt((2 * len(output_df) - 1) / len(output_df))
    p_value_test = 1 - norm.cdf(z_score_test)
    print("Z-score:", z_score_test)
    f.write("\nZ-score (Test Set): %s" % z_score_test)
    print("P-value:", p_value_test)
    f.write("\nP-value (Test Set): %s" % p_value_test)
        
        
    # Compute average tnr, fpr, and fnr scores across all folds
    mean_tpr = np.mean(tpr_list)
    mean_tnr = np.mean(tnr_list)
    mean_fnr = np.mean(fnr_list)
        
    mean_fpr_list = np.mean(fpr_list)
        
    # Compute the average precision and recall and its AUC from all folds
    mean_p_r_auc = plot_precision_recall_curve(precision_test_list, mean_recall, p_r_auc_list, algorithm, algorithm_folder)
        
    # Compute average test accuracy scores across all folds
    test_acc = np.mean(test_accurcy_list)
    f.write("\nAverage Test Accuracy Score: %s" % test_acc)
    print("Average Test Accuracy Score:", test_acc)
    test_dummy = np.mean(test_dummy_list)
        
    # Plot the average AUC score for this algorithm for test set
    plot_auc_scores_fold(algorithm, roc_auc_list_test, mean_auc_test, algorithm_folder)
        
    # Plot the average roc_curve for this algorithm
    mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u = plot_mean_roc_curve(algorithm, mean_fpr, mean_tpr_train, mean_tpr_test, mean_auc_train, mean_auc_test, tprs_lower_train, tprs_upper_train, 
                                      tprs_lower_test, tprs_upper_test, conf_int_train, conf_int_test, algorithm_folder, options['Youden'])
    plt.close()
        
    # Plot the average test accuracy score for this algorithm
    plot_test_accuracy(algorithm, test_accurcy_list, test_acc, test_dummy_list, test_dummy, algorithm_folder)
        
        
    # calcaute the average stat values
    mean_ppv = np.mean(ppv_list)
    mean_npv = np.mean(npv_list)
    f1_mean = np.mean(f1_list)
    dor_mean = np.mean(dor_list)
        
    # get full performance metrics
    save_performance_metrics(algorithm_folder, algorithm, mean_tpr, mean_fpr_list, mean_tnr, 
                                 mean_fnr, mean_ppv, mean_npv, f1_mean, dor_mean, z_score_test, p_value_test, 
                             mean_auc_test, test_acc)
        
    # Divide summed confusion matrix by the number of folds to get the average
    if options['confusion_matrix'] != "None":
        avg_conf_matrix = sum_conf_matrix / options['CV']
        
    # Plot and save a .png file the average confusion matrix for the current algorithim
    if options['confusion_matrix'] == 'Rate':
        plot_and_save_confusion_matrix_rate(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on rates (TPR, FPR, etc.)
    elif options['confusion_matrix'] == 'Percentage':
        plot_and_save_confusion_matrix_percentage(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on the number of TP's, FP's, etc. and thier percentage
                
    # Compute averaged SHAP values across folds and plot them
    if options['SHAP'] == "True":
        plot_shap(Common_Tools.resize_arrays_to_smallest(shap_values_list), X_test_fold, output_df, algorithm, algorithm_folder)

    
    f.close()
    
    return mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u

def cross_validation_repeated_stratified_binary(df, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals_raw, experiment_name, project_name):
    '''
    This function is for the repeated stratified cross validation function.
    This is for model evaluation, NOT model creation.
    '''
    #Seperate the inputs and outputs for test data
    input_df, output_df = Common_Tools.split(df, input_cols, label_cols)
    
    # Before the cross-validation loop
    classes = output_df.unique()
    
    print("Classes: ", classes)
    
    algorithm_folder = os.path.join(project_name, experiment_name, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    filename = os.path.join(algorithm_folder, algorithm + ".txt")
    f = open(filename, "w")
    print("Algorithm:", algorithm)
    print("Label:", output_df.name)
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algorithm)
    f.write("\nLabel: %s"% output_df.name)
        
    # Lists to store ROC curve information for each fold
    tpr_list_train = [] # train set
    mean_fpr = np.linspace(1, 0, 100)
    roc_auc_list_train = [] # list of ROC/AUC Scores for each fold
        
    tpr_list_test = [] # tpr for test set
    roc_auc_list_test = [] # list of ROC/AUC Scores for each fold
        
    tpr_list = [] # tpr list
    tnr_list = [] # tnr list
    fnr_list = [] # fnr list
    fpr_list = [] # fpr list
    ppv_list = [] # ppv list
    npv_list = [] # npv for test set
    f1_list = [] # f1 for test set
    dor_list = [] # dor for test set
        
    test_accurcy_list = [] # test accuracy scores
    test_dummy_list = [] # test dummy scores
        
    # Initialize a list to store SHAP values for each fold
    shap_values_list = []
        
    # Initialize lists to store true and predicted probabilities across all folds
    all_prob_true = []
    all_prob_pred = []
        
    # Lists to store P-R curve information for each fold
    precision_test_list = [] # precision test
    recall_test_list = [] # recall test
    p_r_auc_list = [] # auc for precsion and recal curve
    mean_recall = np.linspace(0, 1, 100)
        
    # Inside the cross-validation loop
    sum_conf_matrix = None
        
    # Loop 10 times in StratifiedKFold
    for fold, (train_idx, test_idx) in enumerate(RepeatedStratifiedKFold(n_repeats=options['n_repeats'], n_splits=options['CV'], random_state=42).split(input_df, output_df)):
            
        # get the specified model/estimator
        estimator, param_vals = get_classifier(algorithm, param_vals_raw)
            
        print(fold)
        f.write("\nFold: %s"%fold)
            
        #split training data into train and test sets
        X_train_fold, X_test_fold = input_df.iloc[train_idx], input_df.iloc[test_idx]
        y_train_fold, y_test_fold = output_df.iloc[train_idx], output_df.iloc[test_idx]
            
        X_col = X_train_fold.columns.to_list()
            
        # Imputing Data if chosen
        if options['Impute'] == "True":
            print("Impute")
            imputer = SimpleImputer(strategy = 'mean')
            imputer.fit(X_train_fold)
            X_train_fold = pd.DataFrame(imputer.transform(X_train_fold), columns = X_col)
            #df[numeric_cols] = imputer.transform(df[numeric_cols])
            X_test_fold = pd.DataFrame(imputer.transform(X_test_fold), columns = X_col)
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Scaling if chosen
        if options['Scaling'] == "True":
            X_train_fold, scaler = Common_Tools.scaling(X_train_fold, input_cols, label_cols, numeric_cols, categorical_cols, scalingMethod=options['scalingMethod'])
            X_test_fold[numeric_cols] = scaler.transform(X_test_fold[numeric_cols])
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Rebalance Data if chosen
        if options['rebalance'] == "True":
            print('rebalance')
            X_train_fold, y_train_fold = Common_Tools.rebalance(X_train_fold, y_train_fold, options['rebalance_type'])
        # Feature Selection if chosen
        if options['FeatureSelection'] == "True":
            X_train_fold, selected_features = Common_Tools.feature_selection(X_train_fold, y_train_fold, method=options['method'], type=options['type'], N=options['N_features'], per=options['per'])
            #print(selected_features)
            X_test_fold = X_test_fold[selected_features]
            f.write("\nSelected Features: %s"%selected_features)
        else:
            selected_features = list(X_train_fold.columns)
                
        estimator = Common_Tools.train_tune(estimator, param_vals, X_train_fold, y_train_fold, options['strategy'], itr=options['itr'])
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            estimator = Common_Tools.calibrate(estimator, X_train_fold, y_train_fold)
            
        params = estimator.get_params()
        f.write("\nParameters: %s"%params)
            
        predictions_train = estimator.predict(X_train_fold) # predict with test set
        probas_train = estimator.predict_proba(X_train_fold)
        predictions_test = estimator.predict(X_test_fold) # predict with test set
        probas_test = estimator.predict_proba(X_test_fold) # get probablities with test set
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            prob_true, prob_pred = calibration_curve(y_test_fold, probas_test[:, 1], n_bins=10)
            disp = CalibrationDisplay(prob_true, prob_pred, probas_test[:, 1])
            print(disp)
            disp.plot()
                
            # Append these probabilities to the lists
            print("Prob Sizes: ")
            print(prob_true.shape)
            print(prob_pred.shape)
            all_prob_true.append(prob_true)
            all_prob_pred.append(prob_pred)
            
        train_acc = accuracy_score(y_train_fold, predictions_train) # training accuracy
        test_acc = accuracy_score(y_test_fold, predictions_test) # test accuracy
        test_accurcy_list.append(test_acc)
        if options['dummy_strategy'] == "all zeros":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_zero(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 0's to comapre to trained model
        elif options['dummy_strategy'] == "all ones":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_ones(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 1's to comapre to trained model
        test_dummy_list.append(dumb_acc)
               
        f.write("\nTraining Accuracy: %.2f%%" % (train_acc * 100))
        print("Training Accuracy:", (train_acc * 100))
        f.write("\nTest Accuracy: %.2f%%" % (test_acc * 100))
        print("Test Accuracy:", (test_acc * 100))
        f.write("\nDummy Model (Predict all 0) Accuracy vs. Test: %.2f%%" % (dumb_acc * 100))
        print("Dummy Model (Predict all 0) Accuracy vs. Test:", (dumb_acc * 100))

        # Calculate confusion matrix for this fold
        if options['confusion_matrix'] == 'Rate':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test, normalize='true') # normalize to get tpr, fpr, etc.
        elif options['confusion_matrix'] == 'Percentage':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test) # gets pure number of tp's, fp's, etc.
            
            
        # calculate P and N
        P = np.sum(list(y_test_fold))
        N = len(y_test_fold) - P
            
        # Calculating true negatives, false psotives, false negatives, and true postives
        tn, fp, fn, tp = confusion_matrix(y_test_fold, predictions_test).ravel()
            
        # Calculating TPR (True Positive Rate) and FPR (False Positive Rate) as induivuial values
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
            
        # Calculating TNR (True Negative Rate) and FNR (False Negative Rate) as induivuial values
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
            
        ppv = (tpr * P) / ((tpr * P) + (fpr * N))
        npv = (tnr * N) / ((tnr * N) + (fnr * P))
        f1 = (2.0 * ppv * tpr) / (ppv + tpr)
        dor = (tpr * tnr) / (fpr * fnr)
            
        tpr_list.append(tpr)
        fnr_list.append(fnr)
        tnr_list.append(tnr)
        fpr_list.append(fpr)
        ppv_list.append(ppv)
        npv_list.append(npv)
        f1_list.append(f1)
        dor_list.append(dor)
            
            
        # summed up the confusion matrix
        if options['confusion_matrix'] != "None":
            if sum_conf_matrix is None:
                sum_conf_matrix = conf_matrix
            else:
                sum_conf_matrix += conf_matrix
                
        # Calculate ROC/AUC scores for each fold on train set
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train_fold, probas_train[:, 1])
        roc_auc_train = metrics.auc(fpr_train, tpr_train)
        roc_auc_list_train.append(roc_auc_train)
            
        # Calculate ROC/AUC scores for each fold on test set
        fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test_fold, probas_test[:, 1])
        roc_auc_test = metrics.auc(fpr_test, tpr_test)
        roc_auc_list_test.append(roc_auc_test)
                
        # Interpolate the ROC curve at a common set of points
        tpr_list_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tpr_list_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            
        # calculate fnr_test and tnr_test
        fnr_test = 1.0 - tpr_test
        tnr_test = 1.0 - fpr_test
            
        # calcuate the percision and recall on test set and the AUC for it
        precision_test, recall_test, _ = precision_recall_curve(y_test_fold, probas_test[:, 1])
        p_r_auc = auc(recall_test, precision_test)
        # Add to list
        precision_test_list.append(np.interp(mean_recall, precision_test, recall_test))
        p_r_auc_list.append(p_r_auc)

        filename_prob = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_prob.joblib")
        filename_pred = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_pred.joblib")
        filename_true = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_true_fullset.joblib")
        filename_features = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_selected_features.joblib")

        joblib.dump(probas_test, filename_prob) # test set probablities
        joblib.dump(predictions_test, filename_pred) # test set predictions
        joblib.dump(output_df, filename_true, compress=1) # full set acutal labels
        joblib.dump(selected_features, filename_features, compress=1) # selected features
            
        modelname = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_model.joblib")
        joblib.dump(estimator, modelname)
            
        #""" SHAP Values
        if options['SHAP'] == "True":
            #explaining model
            print(X_test_fold.shape)
            explainer = shap.Explainer(estimator.predict, X_test_fold)
            shap_values_fold = explainer(X_test_fold, max_evals=2000)
            print(shap_values_fold.shape)
                
            shap_values_fold_array = shap_values_fold.values  # Extract SHAP values from Explanation objects
                
            #shap.summary_plot(shap_values_fold_array, X_train_fold, max_display=10, show=True)     
                
            # Append SHAP values to the list
            shap_values_list.append(shap_values_fold_array)
            print("List Shape:",np.shape(shap_values_list))
                
                
        #"""
        f.write('\n')
        
    # Compute average tpr scores across all folds
    mean_tpr_test = np.mean(tpr_list_test, axis=0)
    std_tpr_test = np.std(tpr_list_test, axis=0)
    mean_tpr_train = np.mean(tpr_list_train, axis=0)
    std_tpr_train = np.std(tpr_list_train, axis=0)

    # Compute average ROC/AUC scores across all folds for train set
    #mean_auc_train = np.mean(roc_auc_list_train) # train set
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    std_auc_train = np.nanstd(roc_auc_list_train) # std
        
    # get confidence interval for test AUC score for train set
    se_tpr_train = std_tpr_train/np.sqrt(options['CV'])
    se_auc_train = std_auc_train/np.sqrt(options['CV'])
    t_score_train = stats.t.ppf(1-0.025, options['CV']) # t-score
    tprs_upper_train = np.minimum(mean_tpr_train + t_score_train*se_tpr_train, 1)
    tprs_lower_train = np.maximum(mean_tpr_train - t_score_train*se_tpr_train, 0)
    cilow = np.maximum(mean_auc_train-t_score_train*se_auc_train, 0) # lower bound
    cihigh = np.minimum(mean_auc_train+t_score_train*se_auc_train, 1) # upper bound
    conf_int_train = [cilow, cihigh]
        
    f.write("\nAverage AUC Score (Train Set): %s" % mean_auc_train)
    print("Average AUC Score (Train Set):", mean_auc_train)
    f.write("\n95%% Confidence Interval (Train Set): [%0.2f, %0.2f]" % (conf_int_train[0], conf_int_train[1]))
    print("95% Confidence Interval (Train Set):", conf_int_train)

    # Compute average ROC/AUC scores across all folds for test set
    mean_auc_test = auc(mean_fpr, mean_tpr_test) # test set
    std_auc_test = np.nanstd(roc_auc_list_test) # std
    
    # get confidence interval for test AUC score for test set
    se_tpr_test = std_tpr_test/np.sqrt(options['CV'])
    se_auc_test = std_auc_test/np.sqrt(options['CV'])
    t_score_test = stats.t.ppf(1-0.025, options['CV'])
    tprs_upper_test = np.minimum(mean_tpr_test + t_score_test*se_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - t_score_test*se_tpr_test, 0) 
    cilow = np.maximum(mean_auc_test-t_score_test*se_auc_test, 0) # lower bound
    cihigh = np.minimum(mean_auc_test+t_score_test*se_auc_test, 1) # upper bound
    conf_int_test = [cilow, cihigh]
        
    f.write("\nAverage AUC Score (Test Set): %s" % mean_auc_test)
    print("Average AUC Score (Test Set):", mean_auc_test)
    f.write("\n95%% Confidence Interval (Test Set): [%0.2f, %0.2f]" % (conf_int_test[0], conf_int_test[1]))
    print("95% Confidence Interval (Test Set):", conf_int_test)

    auc_null = 0.5  # Null hypothesis: AUC = 0.5
    z_score_test = mean_auc_test / np.sqrt((2 * len(output_df) - 1) / len(output_df))
    p_value_test = 1 - norm.cdf(z_score_test)
    print("Z-score:", z_score_test)
    f.write("\nZ-score (Test Set): %s" % z_score_test)
    print("P-value:", p_value_test)
    f.write("\nP-value (Test Set): %s" % p_value_test)
        
        
    # Compute average tnr, fpr, and fnr scores across all folds
    mean_tpr = np.mean(tpr_list)
    mean_tnr = np.mean(tnr_list)
    mean_fnr = np.mean(fnr_list)
        
    mean_fpr_list = np.mean(fpr_list)
        
    # Compute the average precision and recall and its AUC from all folds
    mean_p_r_auc = plot_precision_recall_curve(precision_test_list, mean_recall, p_r_auc_list, algorithm, algorithm_folder)
        
    # Compute average test accuracy scores across all folds
    test_acc = np.mean(test_accurcy_list)
    f.write("\nAverage Test Accuracy Score: %s" % test_acc)
    print("Average Test Accuracy Score:", test_acc)
    test_dummy = np.mean(test_dummy_list)
        
    # Plot the average AUC score for this algorithm for test set
    plot_auc_scores_fold(algorithm, roc_auc_list_test, mean_auc_test, algorithm_folder)
        
    # Plot the average roc_curve for this algorithm
    mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u = plot_mean_roc_curve(algorithm, mean_fpr, mean_tpr_train, mean_tpr_test, mean_auc_train, mean_auc_test, tprs_lower_train, tprs_upper_train, 
                                      tprs_lower_test, tprs_upper_test, conf_int_train, conf_int_test, algorithm_folder, options['Youden'])
    plt.close()
        
    # Plot the average test accuracy score for this algorithm
    plot_test_accuracy(algorithm, test_accurcy_list, test_acc, test_dummy_list, test_dummy, algorithm_folder)
        
        
    # calcaute the average stat values
    mean_ppv = np.mean(ppv_list)
    mean_npv = np.mean(npv_list)
    f1_mean = np.mean(f1_list)
    dor_mean = np.mean(dor_list)
        
    # get full performance metrics
    save_performance_metrics(algorithm_folder, algorithm, mean_tpr, mean_fpr_list, mean_tnr, 
                                 mean_fnr, mean_ppv, mean_npv, f1_mean, dor_mean, z_score_test, p_value_test, 
                             mean_auc_test, test_acc)
        
    # Divide summed confusion matrix by the number of folds to get the average
    if options['confusion_matrix'] != "None":
        avg_conf_matrix = sum_conf_matrix / options['CV']
        
    # Plot and save a .png file the average confusion matrix for the current algorithim
    if options['confusion_matrix'] == 'Rate':
        plot_and_save_confusion_matrix_rate(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on rates (TPR, FPR, etc.)
    elif options['confusion_matrix'] == 'Percentage':
        plot_and_save_confusion_matrix_percentage(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on the number of TP's, FP's, etc. and thier percentage
                
    # Compute averaged SHAP values across folds and plot them
    if options['SHAP'] == "True":
        plot_shap(Common_Tools.resize_arrays_to_smallest(shap_values_list), X_test_fold, output_df, algorithm, algorithm_folder)

    
    f.close()
    
    return mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u

def cross_validation_binary(df, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals_raw, experiment_name, project_name):
    '''
    This function is for the regular cross validation (without any stratifation) function.
    This is for model evaluation, NOT model creation.
    '''
    #Seperate the inputs and outputs for test data
    input_df, output_df = Common_Tools.split(df, input_cols, label_cols)
    
    # Before the cross-validation loop
    classes = output_df.unique()
    
    print("Classes: ", classes)
    
    algorithm_folder = os.path.join(project_name, experiment_name, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    filename = os.path.join(algorithm_folder, algorithm + ".txt")
    f = open(filename, "w")
    print("Algorithm:", algorithm)
    print("Label:", output_df.name)
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algorithm)
    f.write("\nLabel: %s"% output_df.name)
        
    # Lists to store ROC curve information for each fold
    tpr_list_train = [] # train set
    mean_fpr = np.linspace(1, 0, 100)
    roc_auc_list_train = [] # list of ROC/AUC Scores for each fold
        
    tpr_list_test = [] # tpr for test set
    roc_auc_list_test = [] # list of ROC/AUC Scores for each fold
        
    tpr_list = [] # tpr list
    tnr_list = [] # tnr list
    fnr_list = [] # fnr list
    fpr_list = [] # fpr list
    ppv_list = [] # ppv list
    npv_list = [] # npv for test set
    f1_list = [] # f1 for test set
    dor_list = [] # dor for test set
        
    test_accurcy_list = [] # test accuracy scores
    test_dummy_list = [] # test dummy scores
        
    # Initialize a list to store SHAP values for each fold
    shap_values_list = []
        
    # Initialize lists to store true and predicted probabilities across all folds
    all_prob_true = []
    all_prob_pred = []
        
    # Lists to store P-R curve information for each fold
    precision_test_list = [] # precision test
    recall_test_list = [] # recall test
    p_r_auc_list = [] # auc for precsion and recal curve
    mean_recall = np.linspace(0, 1, 100)
        
    # Inside the cross-validation loop
    sum_conf_matrix = None
        
    # Loop 10 times in KFold
    for fold, (train_idx, test_idx) in enumerate(KFold(n_splits=options['CV'], shuffle=True, random_state=42).split(input_df, output_df)):
            
        # get the specified model/estimator
        estimator, param_vals = get_classifier(algorithm, param_vals_raw)
            
        print(fold)
        f.write("\nFold: %s"%fold)
            
        #split training data into train and test sets
        X_train_fold, X_test_fold = input_df.iloc[train_idx], input_df.iloc[test_idx]
        y_train_fold, y_test_fold = output_df.iloc[train_idx], output_df.iloc[test_idx]
            
        X_col = X_train_fold.columns.to_list()
            
        # Imputing Data if chosen
        if options['Impute'] == "True":
            print("Impute")
            imputer = SimpleImputer(strategy = 'mean')
            imputer.fit(X_train_fold)
            X_train_fold = pd.DataFrame(imputer.transform(X_train_fold), columns = X_col)
            #df[numeric_cols] = imputer.transform(df[numeric_cols])
            X_test_fold = pd.DataFrame(imputer.transform(X_test_fold), columns = X_col)
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Scaling if chosen
        if options['Scaling'] == "True":
            X_train_fold, scaler = Common_Tools.scaling(X_train_fold, input_cols, label_cols, numeric_cols, categorical_cols, scalingMethod=options['scalingMethod'])
            X_test_fold[numeric_cols] = scaler.transform(X_test_fold[numeric_cols])
            y_train_fold.reset_index(drop=True, inplace=True)
                
        # Rebalance Data if chosen
        if options['rebalance'] == "True":
            print('rebalance')
            X_train_fold, y_train_fold = Common_Tools.rebalance(X_train_fold, y_train_fold, options['rebalance_type'])
        # Feature Selection if chosen
        if options['FeatureSelection'] == "True":
            X_train_fold, selected_features = Common_Tools.feature_selection(X_train_fold, y_train_fold, method=options['method'], type=options['type'], N=options['N_features'], per=options['per'])
            #print(selected_features)
            X_test_fold = X_test_fold[selected_features]
            f.write("\nSelected Features: %s"%selected_features)
        else:
            selected_features = list(X_train_fold.columns)
                
        estimator = Common_Tools.train_tune(estimator, param_vals, X_train_fold, y_train_fold, options['strategy'], itr=options['itr'])
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            estimator = Common_Tools.calibrate(estimator, X_train_fold, y_train_fold)
            
        params = estimator.get_params()
        f.write("\nParameters: %s"%params)
            
        predictions_train = estimator.predict(X_train_fold) # predict with test set
        probas_train = estimator.predict_proba(X_train_fold)
        predictions_test = estimator.predict(X_test_fold) # predict with test set
        probas_test = estimator.predict_proba(X_test_fold) # get probablities with test set
            
        # Calibrate the model if chosen
        if options['calibrate'] == "True":
            prob_true, prob_pred = calibration_curve(y_test_fold, probas_test[:, 1], n_bins=10)
            disp = CalibrationDisplay(prob_true, prob_pred, probas_test[:, 1])
            print(disp)
            disp.plot()
                
            # Append these probabilities to the lists
            print("Prob Sizes: ")
            print(prob_true.shape)
            print(prob_pred.shape)
            all_prob_true.append(prob_true)
            all_prob_pred.append(prob_pred)
            
        train_acc = accuracy_score(y_train_fold, predictions_train) # training accuracy
        test_acc = accuracy_score(y_test_fold, predictions_test) # test accuracy
        test_accurcy_list.append(test_acc)
        if options['dummy_strategy'] == "all zeros":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_zero(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 0's to comapre to trained model
        elif options['dummy_strategy'] == "all ones":
            dumb_acc = accuracy_score(y_test_fold, Common_Tools.all_ones(X_test_fold, y_test_fold)) # use a dummy model  where it predicts all 1's to comapre to trained model
        test_dummy_list.append(dumb_acc)
               
        f.write("\nTraining Accuracy: %.2f%%" % (train_acc * 100))
        print("Training Accuracy:", (train_acc * 100))
        f.write("\nTest Accuracy: %.2f%%" % (test_acc * 100))
        print("Test Accuracy:", (test_acc * 100))
        f.write("\nDummy Model (Predict all 0) Accuracy vs. Test: %.2f%%" % (dumb_acc * 100))
        print("Dummy Model (Predict all 0) Accuracy vs. Test:", (dumb_acc * 100))

        # Calculate confusion matrix for this fold
        if options['confusion_matrix'] == 'Rate':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test, normalize='true') # normalize to get tpr, fpr, etc.
        elif options['confusion_matrix'] == 'Percentage':
            conf_matrix = confusion_matrix(y_test_fold, predictions_test) # gets pure number of tp's, fp's, etc.
            
            
        # calculate P and N
        P = np.sum(list(y_test_fold))
        N = len(y_test_fold) - P
            
        # Calculating true negatives, false psotives, false negatives, and true postives
        tn, fp, fn, tp = confusion_matrix(y_test_fold, predictions_test).ravel()
            
        # Calculating TPR (True Positive Rate) and FPR (False Positive Rate) as induivuial values
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
            
        # Calculating TNR (True Negative Rate) and FNR (False Negative Rate) as induivuial values
        tnr = tn / (tn + fp)
        fnr = fn / (fn + tp)
            
        ppv = (tpr * P) / ((tpr * P) + (fpr * N))
        npv = (tnr * N) / ((tnr * N) + (fnr * P))
        f1 = (2.0 * ppv * tpr) / (ppv + tpr)
        dor = (tpr * tnr) / (fpr * fnr)
            
        tpr_list.append(tpr)
        fnr_list.append(fnr)
        tnr_list.append(tnr)
        fpr_list.append(fpr)
        ppv_list.append(ppv)
        npv_list.append(npv)
        f1_list.append(f1)
        dor_list.append(dor)
            
            
        # summed up the confusion matrix
        if options['confusion_matrix'] != "None":
            if sum_conf_matrix is None:
                sum_conf_matrix = conf_matrix
            else:
                sum_conf_matrix += conf_matrix
                
        # Calculate ROC/AUC scores for each fold on train set
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train_fold, probas_train[:, 1])
        roc_auc_train = metrics.auc(fpr_train, tpr_train)
        roc_auc_list_train.append(roc_auc_train)
            
        # Calculate ROC/AUC scores for each fold on test set
        fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test_fold, probas_test[:, 1])
        roc_auc_test = metrics.auc(fpr_test, tpr_test)
        roc_auc_list_test.append(roc_auc_test)
                
        # Interpolate the ROC curve at a common set of points
        tpr_list_test.append(np.interp(mean_fpr, fpr_test, tpr_test))
        tpr_list_train.append(np.interp(mean_fpr, fpr_train, tpr_train))
            
        # calculate fnr_test and tnr_test
        fnr_test = 1.0 - tpr_test
        tnr_test = 1.0 - fpr_test
            
        # calcuate the percision and recall on test set and the AUC for it
        precision_test, recall_test, _ = precision_recall_curve(y_test_fold, probas_test[:, 1])
        p_r_auc = auc(recall_test, precision_test)
        # Add to list
        precision_test_list.append(np.interp(mean_recall, precision_test, recall_test))
        p_r_auc_list.append(p_r_auc)

        filename_prob = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_prob.joblib")
        filename_pred = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_pred.joblib")
        filename_true = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_true_fullset.joblib")
        filename_features = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_selected_features.joblib")

        joblib.dump(probas_test, filename_prob) # test set probablities
        joblib.dump(predictions_test, filename_pred) # test set predictions
        joblib.dump(output_df, filename_true, compress=1) # full set acutal labels
        joblib.dump(selected_features, filename_features, compress=1) # selected features
            
        modelname = os.path.join(algorithm_folder, algorithm + "_" + output_df.name + "_model.joblib")
        joblib.dump(estimator, modelname)
            
        #""" SHAP Values
        if options['SHAP'] == "True":
            #explaining model
            print(X_test_fold.shape)
            explainer = shap.Explainer(estimator.predict, X_test_fold)
            shap_values_fold = explainer(X_test_fold, max_evals=2000)
            print(shap_values_fold.shape)
                
            shap_values_fold_array = shap_values_fold.values  # Extract SHAP values from Explanation objects
                
            #shap.summary_plot(shap_values_fold_array, X_train_fold, max_display=10, show=True)     
                
            # Append SHAP values to the list
            shap_values_list.append(shap_values_fold_array)
            print("List Shape:",np.shape(shap_values_list))
                
                
        #"""
        f.write('\n')
        
    # Compute average tpr scores across all folds
    mean_tpr_test = np.mean(tpr_list_test, axis=0)
    std_tpr_test = np.std(tpr_list_test, axis=0)
    mean_tpr_train = np.mean(tpr_list_train, axis=0)
    std_tpr_train = np.std(tpr_list_train, axis=0)

    # Compute average ROC/AUC scores across all folds for train set
    #mean_auc_train = np.mean(roc_auc_list_train) # train set
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    std_auc_train = np.nanstd(roc_auc_list_train) # std
        
    # get confidence interval for test AUC score for train set
    se_tpr_train = std_tpr_train/np.sqrt(options['CV'])
    se_auc_train = std_auc_train/np.sqrt(options['CV'])
    t_score_train = stats.t.ppf(1-0.025, options['CV']) # t-score
    tprs_upper_train = np.minimum(mean_tpr_train + t_score_train*se_tpr_train, 1)
    tprs_lower_train = np.maximum(mean_tpr_train - t_score_train*se_tpr_train, 0)
    cilow = np.maximum(mean_auc_train-t_score_train*se_auc_train, 0) # lower bound
    cihigh = np.minimum(mean_auc_train+t_score_train*se_auc_train, 1) # upper bound
    conf_int_train = [cilow, cihigh]
        
    f.write("\nAverage AUC Score (Train Set): %s" % mean_auc_train)
    print("Average AUC Score (Train Set):", mean_auc_train)
    f.write("\n95%% Confidence Interval (Train Set): [%0.2f, %0.2f]" % (conf_int_train[0], conf_int_train[1]))
    print("95% Confidence Interval (Train Set):", conf_int_train)

    # Compute average ROC/AUC scores across all folds for test set
    mean_auc_test = auc(mean_fpr, mean_tpr_test) # test set
    std_auc_test = np.nanstd(roc_auc_list_test) # std
    
    # get confidence interval for test AUC score for test set
    se_tpr_test = std_tpr_test/np.sqrt(options['CV'])
    se_auc_test = std_auc_test/np.sqrt(options['CV'])
    t_score_test = stats.t.ppf(1-0.025, options['CV'])
    tprs_upper_test = np.minimum(mean_tpr_test + t_score_test*se_tpr_test, 1)
    tprs_lower_test = np.maximum(mean_tpr_test - t_score_test*se_tpr_test, 0) 
    cilow = np.maximum(mean_auc_test-t_score_test*se_auc_test, 0) # lower bound
    cihigh = np.minimum(mean_auc_test+t_score_test*se_auc_test, 1) # upper bound
    conf_int_test = [cilow, cihigh]
        
    f.write("\nAverage AUC Score (Test Set): %s" % mean_auc_test)
    print("Average AUC Score (Test Set):", mean_auc_test)
    f.write("\n95%% Confidence Interval (Test Set): [%0.2f, %0.2f]" % (conf_int_test[0], conf_int_test[1]))
    print("95% Confidence Interval (Test Set):", conf_int_test)

    auc_null = 0.5  # Null hypothesis: AUC = 0.5
    z_score_test = mean_auc_test / np.sqrt((2 * len(output_df) - 1) / len(output_df))
    p_value_test = 1 - norm.cdf(z_score_test)
    print("Z-score:", z_score_test)
    f.write("\nZ-score (Test Set): %s" % z_score_test)
    print("P-value:", p_value_test)
    f.write("\nP-value (Test Set): %s" % p_value_test)
        
        
    # Compute average tnr, fpr, and fnr scores across all folds
    mean_tpr = np.mean(tpr_list)
    mean_tnr = np.mean(tnr_list)
    mean_fnr = np.mean(fnr_list)
        
    mean_fpr_list = np.mean(fpr_list)
        
    # Compute the average precision and recall and its AUC from all folds
    mean_p_r_auc = plot_precision_recall_curve(precision_test_list, mean_recall, p_r_auc_list, algorithm, algorithm_folder)
        
    # Compute average test accuracy scores across all folds
    test_acc = np.mean(test_accurcy_list)
    f.write("\nAverage Test Accuracy Score: %s" % test_acc)
    print("Average Test Accuracy Score:", test_acc)
    test_dummy = np.mean(test_dummy_list)
        
    # Plot the average AUC score for this algorithm for test set
    plot_auc_scores_fold(algorithm, roc_auc_list_test, mean_auc_test, algorithm_folder)
        
    # Plot the average roc_curve for this algorithm
    mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u = plot_mean_roc_curve(algorithm, mean_fpr, mean_tpr_train, mean_tpr_test, mean_auc_train, mean_auc_test, tprs_lower_train, tprs_upper_train, 
                                      tprs_lower_test, tprs_upper_test, conf_int_train, conf_int_test, algorithm_folder, options['Youden'])
    plt.close()
        
    # Plot the average test accuracy score for this algorithm
    plot_test_accuracy(algorithm, test_accurcy_list, test_acc, test_dummy_list, test_dummy, algorithm_folder)
        
        
    # calcaute the average stat values
    mean_ppv = np.mean(ppv_list)
    mean_npv = np.mean(npv_list)
    f1_mean = np.mean(f1_list)
    dor_mean = np.mean(dor_list)
        
    # get full performance metrics
    save_performance_metrics(algorithm_folder, algorithm, mean_tpr, mean_fpr_list, mean_tnr, 
                                 mean_fnr, mean_ppv, mean_npv, f1_mean, dor_mean, z_score_test, p_value_test, 
                             mean_auc_test, test_acc)
        
    # Divide summed confusion matrix by the number of folds to get the average
    if options['confusion_matrix'] != "None":
        avg_conf_matrix = sum_conf_matrix / options['CV']
        
    # Plot and save a .png file the average confusion matrix for the current algorithim
    if options['confusion_matrix'] == 'Rate':
        plot_and_save_confusion_matrix_rate(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on rates (TPR, FPR, etc.)
    elif options['confusion_matrix'] == 'Percentage':
        plot_and_save_confusion_matrix_percentage(avg_conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on the number of TP's, FP's, etc. and thier percentage
                
    # Compute averaged SHAP values across folds and plot them
    if options['SHAP'] == "True":
        plot_shap(Common_Tools.resize_arrays_to_smallest(shap_values_list), X_test_fold, output_df, algorithm, algorithm_folder)

    
    f.close()
    
    return mean_fpr, mean_tpr_test, mean_auc_test, cd_interval_l, cd_interval_u

def train_test_evaluation_binary(df_train, df_test, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals_raw, experiment_name, project_name):
    '''
    This function is for seperate training and test sets.
    '''
    #Seperate the inputs and outputs for test data
    X_train, y_train = Common_Tools.split(df_train, input_cols, label_cols)
    
    X_test, y_test = Common_Tools.split(df_test, input_cols, label_cols)
    
    # Before the cross-validation loop
    classes = y_test.unique()
    
    print("Classes: ", classes)
    
    algorithm_folder = os.path.join(project_name, experiment_name, algorithm)
    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
    filename = os.path.join(algorithm_folder, algorithm + ".txt")
    f = open(filename, "w")
    print("Algorithm:", algorithm)
    print("Label:", y_test.name)
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algorithm)
    f.write("\nLabel: %s"% y_test.name)
        
    # get the specified model/estimator
    estimator, param_vals = get_classifier(algorithm, param_vals_raw)
            
    X_col = X_train.columns.to_list()
            
    # Imputing Data if chosen
    if options['Impute'] == "True":
        print("Impute")
        imputer = SimpleImputer(strategy = 'mean')
        imputer.fit(X_train)
        X_train = pd.DataFrame(imputer.transform(X_train), columns = X_col)
        #df[numeric_cols] = imputer.transform(df[numeric_cols])
        X_test = pd.DataFrame(imputer.transform(X_test), columns = X_col)
        y_train.reset_index(drop=True, inplace=True)
                
    # Scaling if chosen
    if options['Scaling'] == "True":
        X_train, scaler = Common_Tools.scaling(X_train, input_cols, label_cols, numeric_cols, categorical_cols, scalingMethod=options['scalingMethod'])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        y_train.reset_index(drop=True, inplace=True)
                
    # Rebalance Data if chosen
    if options['rebalance'] == "True":
        print('rebalance')
        X_train, y_train = Common_Tools.rebalance(X_train, y_train, options['rebalance_type'])
    # Feature Selection if chosen
    if options['FeatureSelection'] == "True":
        X_train, selected_features = Common_Tools.feature_selection(X_train, y_train, method=options['method'], type=options['type'], N=options['N_features'], per=options['per'])
        #print(selected_features)
        X_test = X_test[selected_features]
        f.write("\nSelected Features: %s"%selected_features)
    else:
        selected_features = list(X_train.columns)
                
    estimator = Common_Tools.train_tune(estimator, param_vals, X_train, y_train, options['strategy'], itr=options['itr'])
            
    # Calibrate the model if chosen
    if options['calibrate'] == "True":
        estimator = Common_Tools.calibrate(estimator, X_train, y_train)
            
    params = estimator.get_params()
    f.write("\nParameters: %s"%params)
            
    predictions_train = estimator.predict(X_train) # predict with test set
    probas_train = estimator.predict_proba(X_train)
    predictions_test = estimator.predict(X_test) # predict with test set
    probas_test = estimator.predict_proba(X_test) # get probablities with test set
            
    # Calibrate the model if chosen
    if options['calibrate'] == "True":
        prob_true, prob_pred = calibration_curve(y_test, probas_test[:, 1], n_bins=10)
        disp = CalibrationDisplay(prob_true, prob_pred, probas_test[:, 1])
        print(disp)
        disp.plot()
                
        # Append these probabilities to the lists
        print("Prob Sizes: ")
        print(prob_true.shape)
        print(prob_pred.shape)
            
    train_acc = accuracy_score(y_train, predictions_train) # training accuracy
    test_acc = accuracy_score(y_test, predictions_test) # test accuracy

    if options['dummy_strategy'] == "all zeros":
        dumb_acc = accuracy_score(y_test, Common_Tools.all_zero(X_test, y_test)) # use a dummy model  where it predicts all 0's to comapre to trained model
    elif options['dummy_strategy'] == "all ones":
        dumb_acc = accuracy_score(y_test, Common_Tools.all_ones(X_test, y_test)) # use a dummy model  where it predicts all 1's to comapre to trained model
               
    f.write("\nTraining Accuracy: %.2f%%" % (train_acc * 100))
    print("Training Accuracy:", (train_acc * 100))
    f.write("\nTest Accuracy: %.2f%%" % (test_acc * 100))
    print("Test Accuracy:", (test_acc * 100))
    f.write("\nDummy Model (Predict all 0) Accuracy vs. Test: %.2f%%" % (dumb_acc * 100))
    print("Dummy Model (Predict all 0) Accuracy vs. Test:", (dumb_acc * 100))

    # Calculate confusion matrix
    if options['confusion_matrix'] == 'Rate':
        conf_matrix = confusion_matrix(y_test, predictions_test, normalize='true') # normalize to get tpr, fpr, etc.
        plot_and_save_confusion_matrix_rate(conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on rates (TPR, FPR, etc.)
    elif options['confusion_matrix'] == 'Percentage':
        conf_matrix = confusion_matrix(y_test, predictions_test) # gets pure number of tp's, fp's, etc.
        plot_and_save_confusion_matrix_percentage(conf_matrix, algorithm, classes, algorithm_folder) # gives the confusion matrix based on the number of TP's, FP's, etc. and thier percentage    
            
    # calculate P and N
    P = np.sum(list(y_test))
    N = len(y_test) - P
            
    # Calculating true negatives, false psotives, false negatives, and true postives
    tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
            
    # Calculating TPR (True Positive Rate) and FPR (False Positive Rate) as induivuial values
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
            
    # Calculating TNR (True Negative Rate) and FNR (False Negative Rate) as induivuial values
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
            
    ppv = (tpr * P) / ((tpr * P) + (fpr * N))
    npv = (tnr * N) / ((tnr * N) + (fnr * P))
    f1 = (2.0 * ppv * tpr) / (ppv + tpr)
    dor = (tpr * tnr) / (fpr * fnr)
    
    youden_index = (tp / (tp + fn)) + (tn / (fp + tn)) - 1
                
    fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = roc_auc_curve(y_train, probas_train, y_test, probas_test, algorithm, algorithm_folder, options['Youden'],  youden_index)                           
            
    p_r_curve(y_test, probas_test, algorithm, algorithm_folder)

    filename_prob = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_prob.joblib")
    filename_pred = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_pred.joblib")
    filename_true = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_true_fullset.joblib")
    filename_features = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_selected_features.joblib")

    joblib.dump(probas_test, filename_prob) # test set probablities
    joblib.dump(predictions_test, filename_pred) # test set predictions
    joblib.dump([y_train, y_test], filename_true, compress=1) # full set acutal labels
    joblib.dump(selected_features, filename_features, compress=1) # selected features
            
    modelname = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_model.joblib")
    joblib.dump(estimator, modelname)
            
    #""" SHAP Values
    if options['SHAP'] == "True":
        #explaining model
        print(X_test.shape)
        explainer = shap.Explainer(estimator.predict, X_test)
        shap_values = explainer(X_test, max_evals=2000)
        print(shap_values.shape)
                
        shap_values_array = shap_values.values  # Extract SHAP values from Explanation objects
                
        shap.summary_plot(shap_values_array, X_test, max_display=10, show=False)
        
        filename_shap = os.path.join(algorithm_folder, algorithm + "_" + y_test.name + "_shap.png")
        plt.savefig(filename_shap,dpi=700)
        plt.show()
                               
        plt.close()
                               
    f.write('\n')
        
    # get full performance metrics
    save_metrics_singluar(tpr, fpr, tnr, fnr, ppv, npv, f1, dor, roc_auc_test, test_acc, youden_index, algorithm, algorithm_folder)

    
    f.close()
    
    return fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u

def plot_all_roc_curves(all_roc_curves, all_experiments, project_name):
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], 'k--')

    for experiment, figure in all_roc_curves.items():

        print(experiment)
        print(figure)

        algorithm = all_experiments[project_name][experiment]['algorithm']
        print(algorithm)

        # Plotting both the curves simultaneously 
        plt.plot(figure['fpr_test'], figure['tpr_test'], label="ROC/AUC Curve (Test Set) for " + experiment + " (" + algorithm + "). AUC={:.3f}, CI: [{:.3f}-{:.3f}]".format(figure['roc_auc_test'], figure['conf_int_test_l'], figure['conf_int_test_u']))

    # Naming the x-axis, y-axis and the whole graph 
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Average AUC for all experiment')

    # Adding legend, which helps us recognize the curve according to it's color 
    plt.legend() 

    output_path = project_name + '/' + 'all_test_auc.png'

    # Save the merged image
    plt.savefig(output_path)

    # To load the display window 
    plt.show()


def binary_run_all_experiments(all_experiments, project_name, unique_value_threshold=10):
    experiment_names = list(all_experiments[project_name].keys())
    print("List of experiments: ", experiment_names)
    
    # all ROC Curves figure for each experiment
    all_roc_curves = {}
    
    for experiment in all_experiments[project_name]:
        experiment_name = experiment
        print(experiment_name)
        
        all_roc_curves[experiment_name] = {}

        set_up = all_experiments[project_name][experiment_name]

        test_set = set_up['test_set']
        
        if test_set == "None":
            print("No Test Set")
            algorithm, df, algorithm_folder, input_cols, label_cols, categorical_cols, numeric_cols, cross_validation, options, param_vals = setup_binary(set_up, experiment_name, project_name, unique_value_threshold=unique_value_threshold)
            df_refined, input_cols, label_cols = Common_Tools.data_prep(df, input_cols, label_cols, numeric_cols, categorical_cols, options)
        else:
            print("Yes Test Set")
            algorithm, df, df_test, algorithm_folder, input_cols, label_cols, categorical_cols, numeric_cols, options, param_vals = setup_test_set_binary(set_up, experiment_name, project_name, unique_value_threshold=unique_value_threshold)
            df_refined, input_cols, label_cols = Common_Tools.data_prep(df, input_cols, label_cols, numeric_cols, categorical_cols, options)
            df_refined_test, _, _ = Common_Tools.data_prep(df_test, input_cols, label_cols, numeric_cols, categorical_cols, options)
        
        algorithm, df, algorithm_folder, input_cols, label_cols, categorical_cols, numeric_cols, cross_validation, options, param_vals = setup_binary(set_up, experiment_name, project_name, unique_value_threshold=unique_value_threshold)
        
        df_refined, input_cols, label_cols = Common_Tools.data_prep(df, input_cols, label_cols, numeric_cols, categorical_cols, options)  
        
        if options['correlation_matrix'] == "True":
            correlation_matrix(df_refined, numeric_cols, label_cols, algorithm_folder)
            top_correlation_variables(df_refined, numeric_cols, label_cols, algorithm_folder)
        
        if test_set != "None":
            print("Test Set Used!")
            fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = train_test_evaluation_binary(df_refined, df_refined_test, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals, experiment_name, project_name)
        else:
            if cross_validation == "Stratified":   
                print("Stratified")
                fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = cross_validation_stratified_binary(df_refined, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals, experiment_name, project_name)
            elif cross_validation == "Repeated Stratified":
                print("Repeated Stratified")
                fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = cross_validation_repeated_stratified_binary(df_refined, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals, experiment_name, project_name)
            elif cross_validation == "Normal":
                print("Normal")
                fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = cross_validation_binary(df_refined, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals, experiment_name, project_name)
            elif cross_validation == "None":
                print("None")
                fpr_test, tpr_test, roc_auc_test, conf_int_test_l, conf_int_test_u = train_test_split_binary(df_refined, input_cols, label_cols, numeric_cols, categorical_cols, options, algorithm, param_vals, experiment_name, project_name)
            
        all_roc_curves[experiment_name]['fpr_test'] = fpr_test
        all_roc_curves[experiment_name]['tpr_test'] = tpr_test
        all_roc_curves[experiment_name]['roc_auc_test'] = roc_auc_test
        all_roc_curves[experiment_name]['conf_int_test_l'] = conf_int_test_l
        all_roc_curves[experiment_name]['conf_int_test_u'] = conf_int_test_u
        
        print("_____________________________________________________________________________________________________")
  
    #print("ROC Curves: ", all_roc_curves)
    plot_all_roc_curves(all_roc_curves, all_experiments, project_name)  
    
    return all_roc_curves
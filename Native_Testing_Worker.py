import time
import io
import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import csv
#import magic 
import pickle
import random 
from random import randint
from random import uniform
from pathlib import Path
import json
import shap
from scipy import stats
import os
import joblib as joblib
from joblib import dump, load
#np.random.seed(1000)
rstate = 12
import uuid
import streamlit as st
import os
import joblib
import signal
import sys
import traceback
from datetime import datetime
import pprint
import pymongo
from pymongo import MongoClient
from streamlit_cookies_manager import EncryptedCookieManager
from Common_Tools import cleanup_stale_jobs, wrap_text_excel, expand_cell_excel, grid_excel, split, upload_data, save_data, load_data
from roctools import full_roc_curve, plot_roc_curve
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import multiprocessing 
from multiprocessing import Process
from datetime import datetime, timedelta
import threading
from Multi_Outcome_Classification_tools import multi_outcome_hyperparameter_binary, multi_outcome_hyperparameter_binary_train_and_test, multi_outcome_cv
from Common_Tools import sanitize_filename, convert_to_json_compatible, generate_configuration_file, generate_configuration_template, generate_results_table, generate_congfig_file, get_avg_results_dic, wrap_text_excel, expand_cell_excel, grid_excel, generate_all_idx_files, upload_data, load_data, save_data, data_prep, data_prep_train_set, parse_exp_multi_outcomes, setup_multioutcome_binary, refine_binary_outcomes, generate_joblib_model
from roctools import full_roc_curve, plot_roc_curve

def sanitize_filename(filename):
    """Remove or replace invalid characters from filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def find_option_dic(configuration, project_name, algorithm):
    main_dic = configuration[project_name]
    #st.write(main_dic)
    for _, (key, values) in enumerate(main_dic.items()):
        if key == "exp_type" or key == "threshold_type":
            continue
        
        if main_dic[key]["algorithm"] == algorithm:
            return main_dic[key]["options"]

def preprocessdata(df, input_columns, numeric_cols, cutMissingRows='True', threshold=0.75, inf='replace with null', outliers='None', N=20000):
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    if cutMissingRows == 'True':
        print("cutMissingRows")
        # Drop rows with too many missing values
        # computing number of columns
        cols = len(df[input_columns].axes[1])
        print("Cuttoff", int(threshold * cols))
        print(f"Data Size then: {len(df)}")
        df = df.dropna(thresh=int(threshold * cols))
        print(f"Data Size now: {len(df)}")

    if inf == 'replace with null':
        print("replace with null")
        # Replace all inf values with null
        df = df.replace([np.inf, -np.inf], np.nan)
    elif inf == 'replace with zero':
        print("replace with zero")
        # Replace all inf values with null
        df = df.replace([np.inf, -np.inf], 0)

    # Outlier handling
    if outliers == 'remove rows':
        print("remove rows")
        # Remove rows that have a value greater than N for any column. Default N is 20000
        mask = (df[numeric_cols] > N).any(axis=1)
        df = df.loc[~mask]
    elif outliers == 'log':
        print("log")
        # Log values that are greater than N for any column. Default N is 20000
        df.loc[:, numeric_cols] = df[numeric_cols].apply(lambda x: np.where(x > N, np.log(x), x))
        
    # Enforce numeric dtypes for safety
    df.loc[:, numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return df

# plot the ROC Curve
def plot_roc(res, res_array, y_test, probas_test, algo_name, outcome_name, algorithm_folder):
    #with mpl_lock:
    fig, ax = plt.subplots(figsize=(12, 8))
                
    # calcaute the AUROC
    fpr, tpr, _ = roc_curve(y_test, probas_test[:, 1])
    roc_auc = res['auc']
    auc_ci_low = res['auc_cilow']
    auc_ci_high = res['auc_cihigh']
    specificity = res_array['tnr']
    ax.plot(fpr, tpr, label=f'{outcome_name} (AUC = {roc_auc:.4f} [{auc_ci_low:.4f}, {auc_ci_high:.4f}])', linewidth=2)
                
    # get the CI
    ax.fill_between(1-specificity, res_array['tpr_low'], res_array['tpr_high'], alpha=.2)
                
    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(f"ROC Curve for {outcome_name} on {algo_name}", fontsize=16)
    ax.legend(loc="lower right", fontsize=14)
                
    # save the ROC plot
    filename_roc = os.path.join(algorithm_folder, algo_name + "_" + sanitize_filename(outcome_name) + "_roc.png")
    #plt.savefig(filename_roc,dpi=700)
    fig.savefig(filename_roc, dpi=700, bbox_inches="tight")
    #plt.show()  # Display the plot
    plt.close()

def calc_shap(model, X_test, features):
    explainer = shap.Explainer(model.predict, X_test[features])
    #shap_values = explainer.shap_values(X_test[features])
    shap_values = explainer(X_test[features])
    return shap_values

# plot the SHAP Values
def plot_shap(shap_values, X_test, features, algo_name, outcome_name, algorithm_folder):
    #with mpl_lock:
    max_display = min(10, X_test[features].shape[1])

    #shap.initjs()  # harmless for matplotlib plots

    # Create explicit figure + axis FIRST
    fig, ax = plt.subplots(figsize=(10, 6))

    # IMPORTANT: do NOT create fig/ax beforehand
    shap.summary_plot(
        shap_values,
        X_test[features],
        plot_type="dot",
        max_display=max_display,
        show=False,
        #color_bar=False  # avoids matplotlib sci() bug
    )

    # Now get the figure SHAP actually used
    #fig = plt.gcf()
    #fig.set_size_inches(10, 6)
    fig.suptitle(f"SHAP Values for {outcome_name} on {algo_name}")

    # Save to disk
    filename_shap = os.path.join(
        algorithm_folder,
        f"{algo_name}_{sanitize_filename(outcome_name)}_shap.png"
    )

    fig.savefig(filename_shap, dpi=700, bbox_inches="tight")
    
    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=700, bbox_inches="tight")
    buf.seek(0)
    image_data = buf.read()

    plt.close(fig)

    return image_data

# save metadata
def save_testdata(client_name, exp_name, test_set, test_set_name, data_name_test, uploaded_test_set, data_names_list_test):
    # connect to database
    #client = MongoClient('10.14.1.12', 27017)
    client = MongoClient(client_name, 27017)
    # create the database if it does not already exists
    db = client.machine_learning_database
    # create the results if it does not already exists
    results = db.results
    # create the results if it does not already exists
    datasets = db.datasets

    if results.find_one({"exp_name": exp_name, "test set": test_set_name}) == None:
        # save the test set if it hasn't already
        if data_name_test == None and uploaded_test_set.name not in data_names_list_test:
            st.info(f"Testing Dataset is saving in the database", icon="ℹ️")
            # create a list of ML exp.'s that the dataset was used on
            exp_list = [exp_name]
            #get the current time
            current_datetime = datetime.now()
            current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            # save test set in data folder and database
            os.makedirs("Data Sets", exist_ok=True)
            save_data(uploaded_test_set, test_set, os.path.join("Data Sets", uploaded_test_set))
            dataset_test = {
                    "data_name": uploaded_test_set,
                    "type": "Test",
                    "time_saved": current_time,
                    "data_path": os.path.join("Data Sets", uploaded_test_set),
                    "exps used": exp_list
            }
            datasets.insert_one(dataset_test)
        else:
            st.info(f"Testing Dataset of the same name is already in the database. Will be overwritten in the database", icon="ℹ️")
            test_name = uploaded_test_set if data_name_test == None else data_name_test
            save_data(test_name, test_set, os.path.join("Data Sets", test_name))
            # update the dataset in datbase to trackdown the list of ML exps the set was used on
            dataset = datasets.find_one({"data_name": test_name, "type": "Test"})
            # Get the current list of experiments or initialize it if not present
            exp_list = dataset.get("exps used", [])
            # Add the current project name if it's not already in the list
            if exp_name not in exp_list:
                exp_list.append(exp_name)

            datasets.update_one(
                {"data_name": test_name, "type": "Test"}, # Filter condition
                {"$set": { "exps used": exp_list }} # Update operation
            )
            return "Success"
    else:
        return "Error"    


def generate_results_table(results_dictonary):
    # Initialize an empty list to collect rows
    rows = []
    #no_results = []
    
    for _, (algo, outcomes) in enumerate(results_dictonary.items()):

        for outcome in list(outcomes.keys()):
            print(f'{outcome} for {algo}')
            
            for _, (key, value) in enumerate(outcomes[outcome]['evaluation'].items()): 
                if isinstance(value, np.int32):
                    print("Int 32 Detected!")
                    print(f"It is on {key}")
        
            new_row = {'Outcome': outcome,
                      'Algorithm': algo,
                      'AUROC Score': outcomes[outcome]['evaluation']['AUROC Score'],
                      'AUROC CI Lower': outcomes[outcome]['evaluation']['AUROC CI Low'],
                      'AUROC CI Upper': outcomes[outcome]['evaluation']['AUROC CI High'],
                      'Accuracy': outcomes[outcome]['evaluation']['Accuracy'],
                      'Precision': outcomes[outcome]['evaluation']['precision'],
                      'Recall': outcomes[outcome]['evaluation']['recall'],
                      'F1 Score': outcomes[outcome]['evaluation']['f1 score'],
                      'TPR': outcomes[outcome]['evaluation']['TPR'], # same as Sensitivity 
                      'TNR': outcomes[outcome]['evaluation']['TNR'], # same as Specificity 
                      'FPR': outcomes[outcome]['evaluation']['FPR'], 
                      'FNR': outcomes[outcome]['evaluation']['FNR'],
                      'PPV': outcomes[outcome]['evaluation']['PPV'],
                      'NPV': outcomes[outcome]['evaluation']['NPV'],
                      'TP': outcomes[outcome]['evaluation']['TP'],
                      'FP': outcomes[outcome]['evaluation']['FP'],
                      'TN': outcomes[outcome]['evaluation']['TN'],
                      'FN': outcomes[outcome]['evaluation']['FN'],
                      'Cutoff value': outcomes[outcome]['evaluation']['cutoff'],
                      'P': outcomes[outcome]['evaluation']['P'],
                      'N': outcomes[outcome]['evaluation']['N'],
                      'AUROC Score (Train)': outcomes[outcome]['evaluation']['AUROC Score (Train)'],
                      'AUROC CI Lower (Train)': outcomes[outcome]['evaluation']['AUROC CI Low (Train)'],
                      'AUROC CI Upper (Train)': outcomes[outcome]['evaluation']['AUROC CI High (Train)'],
                      'P (Train)': outcomes[outcome]['evaluation']['P (Train)'],
                      'N (Train)': outcomes[outcome]['evaluation']['N (Train)']}

            
            # Optionally add training confusion matrix values if available
            train_metrics = ['TP (Train)', 'FP (Train)', 'TN (Train)', 'FN (Train)']

            for metric in train_metrics: # add training confusion matrix numbers if they exist. (Train-Test)/Train %
                if metric in list(outcomes[outcome]['evaluation'].keys()):
                    new_row[metric] = outcomes[outcome]['evaluation'][metric]

            # calcuate and add the % change of the train vs. test AUROC score
            new_row['Train vs. Test AUROC change%'] = ((outcomes[outcome]['evaluation']['AUROC Score (Train)'] - outcomes[outcome]['evaluation']['AUROC Score']) / outcomes[outcome]['evaluation']['AUROC Score (Train)']) * 100
            
            # add new row
            rows.append(new_row)

    # Convert the list of rows into a DataFrame
    results_df = pd.DataFrame(rows)
    return results_df
    

def plot_feature_importance(feature_importance_dic, directory_name):
    outcomes = list(feature_importance_dic.keys())
    print(f'Outcomes {outcomes}')
    for outcome in outcomes:
        
        # get the table
        df = feature_importance_dic[outcome]
        
        # the abs value of feature importance values
        df['importance (abs)'] = df['importance'].abs()
        
        # Sort the DataFrame by the column values
        df = df.sort_values(by='importance (abs)', ascending=False)
        
        # plot the 10 top most important features
        plt.figure(figsize=(8, 6))
        ax = df['importance (abs)'][:10].plot(kind='bar')
        plt.title(f'Feature Importance for {outcome} (Top 10)')

        # Add text labels
        for i, v in enumerate(df['importance (abs)'][:10]):
            ax.text(i, v + 0.0005, f'{v:.3f}', ha='center')

        # Set y-axis range from 0 to 1
        ax.set_ylim(0, df['importance (abs)'][0] + 0.01)
        
        # Adjust x-axis labels for readability
        plt.xticks(rotation=45, ha='right')  # Rotate and align right
        
        # Save the plot as a PNG file
        os.makedirs(directory_name, exist_ok=True)
        file_name = f'{directory_name}/{sanitize_filename(outcome)}_feature_importance.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

        plt.close()

        #plt.show()

def generate_results_dictonary(results_list):
    results_dictonary = {}
    for result in results_list:
        algorithm_name = result[0] # get algo name
        outcome_name = result[1] # get outcome name
        metric_dic = result[2] # get metric results
        roc_payload = result[3] # get roc chart
        shap_payload = result[4] # get shap chart

        print("Plotting ROC Charts")
        # plot the ROC Curves
        plot_roc(roc_payload['res'], roc_payload['res_array'], roc_payload['y_test'], roc_payload['probas_test'], roc_payload['algo_name'], roc_payload['outcome_name'], roc_payload['algorithm_folder'])

        print("Plotting SHAP Charts")
        # plot the shap chart for those shap values
        shap_image = plot_shap(shap_payload['shap_values'], shap_payload['X_test'], shap_payload['features'], algorithm_name, outcome_name, shap_payload['algorithm_folder'])
        
        if not results_dictonary or algorithm_name not in results_dictonary: # initalize results_dictonary algorithm_name for if it not already
            results_dictonary[algorithm_name] = {}

        if outcome_name not in results_dictonary[algorithm_name]: # initalize results_dictonary outcome_name for if it not already
            results_dictonary[algorithm_name][outcome_name] = {}

        results_dictonary[algorithm_name][outcome_name]['evaluation'] = metric_dic
        results_dictonary[algorithm_name][outcome_name]['shap values'] = shap_image
    return results_dictonary

# test single model   
def test_model(testing_setup):
    outcome_dic = testing_setup['outcome_dic']
    algorithm_folder = testing_setup['algorithm_folder']
    algo_name = testing_setup['algo_name']
    o_name = testing_setup['o_name']
    options = testing_setup['options']
    input_columns = testing_setup['input_columns']
    cutoff_index = testing_setup['cutoff_index']
    test_data_raw = testing_setup['test_data_raw']


    # create a log file for the outcome testing
    log_filename = os.path.join(algorithm_folder, algo_name + '_' + sanitize_filename(o_name) + "_log.txt")
    f = open(log_filename, "w", encoding="utf-8")
    f.write("_____________________________________________________________________________________________________")
    f.write("\nAlgorithm: %s"% algo_name)
    f.write("\nLabel: %s"% o_name)
    #update_status("running", f"Testing on {o_name} with {algo_name}.") # update status
    metric_dic = {} # dictionary to store all important metrics

    print(o_name)
    keys_list = list(outcome_dic.keys())

    model = outcome_dic['Model']
    f.write("\nModel: %s"% model)
    features = outcome_dic['Features']
    f.write("\nFeatures: %s"% features)
    outcome_name = outcome_dic['Outcome Name']
    df_res_train = outcome_dic['Train Set Res']
    df_array_res_train = outcome_dic['Train Set Res Array']
    print("  Model: ", model)
    print("  Features: ", features)
    print("  Outcome Name: ", outcome_name)
    print("  Res Train Table: ", df_res_train)
    print()
    if model == 'N/A' :
        f.write(f"\n No Model for {outcome_name} for {algo_name}")
        return "Error"
            
    numeric_columns = outcome_dic['Numeric Columns']
    categorical_columns = outcome_dic['Categorical Columns']

    # get a copy of the dataset
    test_data = test_data_raw.copy()

    # Encoder
    if 'Encoder' in keys_list:
        print("Encoder")
        encoder = outcome_dic['Encoder']
        f.write("\nEncoder: %s"% encoder)
        encoded_cols = outcome_dic['Encoded Columns']

        # Convert to string type (to avoid np.isnan on object)
        test_data[categorical_columns] = test_data[categorical_columns].astype(str)
        # Replace placeholders with proper NaN
        test_data[categorical_columns] = test_data[categorical_columns].replace(
            ["nan", "NaN", "None", "NONE", "<NA>", "null", ""], np.nan
        )

        encoded = pd.DataFrame(encoder.transform(test_data[categorical_columns]),
            columns=encoded_cols,
            index=test_data.index
        )
        test_data = test_data.drop(columns=categorical_columns).join(encoded)

    # preprocess the testing data
    test_data = preprocessdata(test_data, input_columns, numeric_columns, cutMissingRows=options['cutMissingRows'], threshold=options['cut threshold'], inf=options['inf'], outliers=options['outliers'], N=options['outliers_N'])

    # Quantile Transformer
    if 'Quantile Transformer' in keys_list:
        print("Quantile Transformer")
        qt = outcome_dic['Quantile Transformer']
        f.write("\nQuantile Transformer: %s"% qt)
        test_data[numeric_columns] = qt.transform(test_data[numeric_columns])
            
                
    #Seperate the inputs and outputs for test data
    try:
        X_test, y_test = split(test_data, input_columns, outcome_name)
        f.write("\nData Size : %s"% len(X_test))
    except:
        print("Input Columns do not match with data set.")
        f.write("Input Columns do not match with data set.")
        return "Error"
    #X_test, y_test = split(test_data, features, outcome_name)
    X_col = X_test.columns.to_list()
                
    # Class label values
    classes = y_test.unique()

    print("Classes: ", classes)
            
    # Count positives and negatives
    positives = np.sum(y_test == 1)  # Count instances of 1
    negatives = np.sum(y_test == 0)  # Count instances of 0

    print("Postive Count (on training set):", positives)
    print("Negative Count (on training set):", negatives)

    if positives <= 0:
        st.error(f"{o_name} has no postive labels. Testing Done for {testing_setup['algo_name'] } on {outcome_name} not possible.")
        f.write(f"{o_name} has no postive labels. Testing Done for {testing_setup['algo_name'] } on {outcome_name} not possible.")
        return "Error"
                
    # Imputing Data
    if 'Imputer' in keys_list:
        print("Impute")
        imputer = outcome_dic['Imputer']
        f.write("\nImputer: %s"% imputer)
        X_test = pd.DataFrame(imputer.transform(X_test), columns = X_col, index=X_test.index)
        y_test.reset_index(drop=True, inplace=True)
    
    # Scaling Data
    if 'Scaler' in keys_list:
        print("Scaling")
        scaler = outcome_dic['Scaler']
        f.write("\nScaler: %s"% scaler)
        #X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
        X_test.loc[:, numeric_columns] = scaler.transform(X_test[numeric_columns])
        y_test.reset_index(drop=True, inplace=True)

    # Normalize the data
    if 'Normalizer' in keys_list:
        print("Normalize")
        normalizer = outcome_dic['Normalizer']
        f.write("\nNormalizer: %s"% normalizer)
        #X_test[numeric_columns] = normalizer.transform(X_test[numeric_columns])
        X_test.loc[:, numeric_columns] = normalizer.transform(X_test[numeric_columns])
        y_test.reset_index(drop=True, inplace=True)


    #print(X_test)
    #print(y_test)
        
    #st.write('Testing Started...')
    # with st.spinner(f"Testing Model for {algo_name} on {outcome_name}..."):
    print(f"Testing Model for {algo_name} on {outcome_name}...")

    probas_test = model.predict_proba(X_test[features]) # get probablities with test set
    #model.predict(X_test[features]) # predict with test set
        
    # create a states able for metric on the test set
    res, res_array = full_roc_curve(y_test, probas_test[:, 1], index=cutoff_index)
    print("Results Array (Test Set): ", res)
    f.write("\nResults Array (Test Set): %s"% res)

    metric_dic['TPR'] = res['tpr']
    metric_dic['TNR'] = res['tnr']
    metric_dic['FPR'] = res['fpr']
    metric_dic['FNR'] = res['fnr']
    metric_dic['PPV'] = res['ppv']
    metric_dic['NPV'] = res['npv']
        
    metric_dic['AUROC Score'] = res['auc']
    metric_dic['AUROC CI Low'] = res['auc_cilow']
    metric_dic['AUROC CI High'] = res['auc_cihigh']
    metric_dic['cutoff type'] = cutoff_index
    metric_dic['cutoff'] = (res['cutoff_mcc'] if cutoff_index=='mcc' else (res['cutoff_ji'] if cutoff_index=='ji' else (res['cutoff_f1'] if cutoff_index=='f1' else res['cutoff_youden']))).astype(float)
        
    print("Cutoff Index: ", metric_dic['cutoff'])

    #st.write(probas_test[:, 1])
    predictions_test = [1 if p >= metric_dic['cutoff'] else 0 for p in probas_test[:, 1]] # predict with test set
    test_acc = accuracy_score(y_test, predictions_test) # test accuracy
    f.write("\nTest Accuracy: %s"% test_acc)
    print("Test Accuracy:", (test_acc * 100))
    metric_dic['Accuracy'] = test_acc
        
    # training set results
    metric_dic['AUROC Score (Train)'] = df_res_train['auc']
    metric_dic['AUROC CI Low (Train)'] = df_res_train['auc_cilow']
    metric_dic['AUROC CI High (Train)'] = df_res_train['auc_cihigh']
    metric_dic['P (Train)'] = df_res_train['P'].astype(float)
    metric_dic['N (Train)'] = df_res_train['N'].astype(float)
    metric_dic['TP (Train)'] = df_res_train['TP']
    metric_dic['FP (Train)'] = df_res_train['FP']
    metric_dic['TN (Train)'] = df_res_train['TN']
    metric_dic['FN (Train)'] = df_res_train['FN']
        
    # Extract TP, FP, TN, FN
    metric_dic['TP'] = res['TP']
    metric_dic['FP'] = res['FP']
    metric_dic['TN'] = res['TN']
    metric_dic['FN'] = res['FN']
        
    metric_dic['P'] = res['P'].astype(float)
    metric_dic['N'] = res['N'].astype(float)
        
    metric_dic['precision'] = res['precision']
    metric_dic['recall'] = res['recall']
    metric_dic['f1 score'] = res['f1 score']
        
    metric_dic['Ground Truths'] = y_test.to_list()
    metric_dic['Predictions'] = predictions_test
    metric_dic['Probability Scores'] = probas_test.tolist()
        
    print(metric_dic)

    # parameters needed to plot roc curves which will be done outside the function
    roc_payload = {
        "res": res,
        "res_array": res_array,
        "y_test": y_test,
        "probas_test": probas_test,
        "algo_name": algo_name,
        "outcome_name": outcome_name,
        "algorithm_folder": algorithm_folder,
    }

    # plot the SHAP Values
    shap_values = calc_shap(model, X_test, features)
    #image_data = plot_shap(shap_values, X_test, features, algo_name, outcome_name, algorithm_folder)
    shap_payload = { # list of items needed to plot shap charts which will be done outside of the worker process
        "shap_values": shap_values,
        "X_test": X_test,
        "features": features,
        "algorithm_folder": algorithm_folder,
    }

    #st.success(f"✅Testing Done for {algo_name} on {outcome_name}!")
    f.write(f"\n✅Testing Done for {algo_name} on {outcome_name}!")
    f.write('\n')
    f.close()

    print("_________________________________________________________")

    #st.success(f"✅Testing for {algo_name} on {outcome_name} is Complete!")
    return (algo_name, outcome_name, metric_dic, roc_payload, shap_payload)  


# testing model function
def test_models(model_dic, configuration, all_algorithms, all_outcomes, input_columns, test_data_raw, project_name, test_set_name, cutoff_index='youden'):  
    results_dictonary = {}
    
    testing_setups = [] # list of parameters that will be passed into test_model function

    for algo_name in all_algorithms:
        print("Algorithm: ", algo_name)
        
        results_dictonary[algo_name] = {}
        
        for o_name in all_outcomes:
            outcome_dic = model_dic[algo_name][o_name]
            #print(type(outcome_dic))
            options = find_option_dic(configuration, project_name, algo_name)
            #st.write(options)
            algorithm_folder = os.path.join("Results", project_name, test_set_name, f'{algo_name} (results)', o_name)
            os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results

            testing_setup = {
                "algo_name": algo_name,
                "o_name": o_name,
                "outcome_dic": outcome_dic,
                "options": options,
                "algorithm_folder": algorithm_folder,
                "project_name": project_name,
                "test_set_name": test_set_name,
                "test_data_raw": test_data_raw,
                "input_columns": input_columns,
                "cutoff_index": cutoff_index, 
                #"client_name": client_name,
                #"job_id": job_id
            }

            testing_setups.append(testing_setup)

    print("Testing Models have started...")
    print("Testing Models...")
    all_results = []
    t2 = time.time()

    all_results = Parallel( # test the models (use Parallel to make the process faster)
        n_jobs=max(1, multiprocessing.cpu_count() - 2),
        backend="loky", # IMPORTANT (default, but explicit is good)
        verbose=10
    )(
        delayed(test_model)(setup) for setup in testing_setups
    )

    t1 = time.time()

    print("Testing Model is Done!")
    print(f'The Time Parallel Computing Took :{t1-t2} seconds')
    #st.write(all_results)

    # save a textfile telling the time it took to test models
    algorithm_folder = os.path.join("Results", project_name, test_set_name)
    filename_time = os.path.join(algorithm_folder, "testing_time.txt")
    f = open(filename_time, "w", encoding="utf-8")
    f.write(f'The Time Parallel Computing Took :{t1-t2} seconds')
    f.close()

    return [r for r in all_results if isinstance(r, tuple)]


def testing_worker(job_id, client_name, exp_name, test_set, test_set_name, data_name_test, uploaded_test_set, models_dic, configuration, selected_algos, selected_outcomes, input_variables, input_cols_og, threshold_type, model_type):
    # connect to database
    #client = MongoClient('10.14.1.12', 27017)
    client = MongoClient(client_name, 27017)
    # create the database if it does not already exists
    db = client.machine_learning_database
    # create tables for models in the databse
    models = db.models
    # create the results if it does not already exists
    results = db.results
    # create the results if it does not already exists
    datasets = db.datasets
    # create the jobs if it does not already exists
    jobs = db.jobs
    # get all unique exp. names from results collection
    exp_names = db.models.distinct("exp_name", {"type": "Native"})
    # get all training data names from database
    data_names_train = db.datasets.distinct("data_name", {"type": "Train"})
    # get all testing data names from database
    data_names_list_test = db.datasets.distinct("data_name", {"type": "Test"})

    # create a jobs dictonary to store in the MongoDB database
    job_doc = {
        "job_id": job_id,
        "job_type": "Testing (Sklearn)",
        "exp_name": exp_name,
        "status": "queued",
        "message": "Waiting to start...",
        "created_at": datetime.now(),
        "last_updated": datetime.now()
    }
    db.jobs.insert_one(job_doc)

    def update_status(state, message): # inner function to update the status of the current job
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": state,
                "message": message,
                "last_updated": datetime.now()
            }})
        
    def update_status_when_done(state, message): # inner function to update the status when it is finsihed
        completed_time = datetime.now()
        expires_at = completed_time + timedelta(days=1)
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": state,
                "message": message,
                "last_updated": completed_time,
                "completed_at": completed_time,
                "expires_at": expires_at
            }})
        
    def heartbeat():
        jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "heartbeat": datetime.now(),
                "last_updated": datetime.now()
            }}
        )

    stop_heartbeat = False
    def heartbeat_loop():
        while not stop_heartbeat:
            heartbeat()
            time.sleep(30)

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        daemon=True
    )
    heartbeat_thread.start()
        
    
    def handle_termination(signum, frame):
        jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "Stopped",
                "message": "Job was terminated unexpectedly.",
                "last_updated": datetime.now(),
                "expires_at": datetime.now() + timedelta(days=1)
            }}
        )
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGINT, handle_termination)
    
    try:
        update_status("running", "Saving test data...")
        is_success = save_testdata(client_name, exp_name, test_set, test_set_name, data_name_test, uploaded_test_set, data_names_list_test) # save the testing data into database first
            
        if is_success == "Success":
            update_status("running", "Model testing in progress...")
            # call the function to test models
            results_list = test_models(models_dic, configuration, selected_algos, selected_outcomes, input_variables, test_set, exp_name, test_set_name, cutoff_index=threshold_type)
            update_status("running", "Testing Done, Now saving results...")

            if results_list == "Error": # if test_models function returns an error
                print(f"Testing '{exp_name}' Failed. Go back and check testing configuration.")
                jobs.update_one(
                    {"job_id": job_id},
                    {"$set": {
                        "status": "Failed",
                        "message": f"Testing '{exp_name}' Failed. Go back and check testing configuration.",
                        "last_updated": datetime.now(),
                        "expires_at": datetime.now() + timedelta(days=1)
                    }}
                )
                
            else:
                update_status("running", "Saving results...")
                # if no error, then save the results
                algorithm_folder = os.path.join("Results", exp_name, test_set_name)
                os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
                filename = os.path.join(algorithm_folder, "metadata.txt")
                f = open(filename, "w", encoding="utf-8")
                f.write("\nExp Name: %s"% exp_name)
                f.write("\nInput Columns: %s"% input_cols_og)
                f.write("\nOutput Columns: %s"% selected_outcomes)
                f.write("\nAlgorithms: %s"% selected_algos)
                f.close()

                results_dictonary = generate_results_dictonary(results_list) # generate a well orginized results dictonary (and also plot ROC and SHAP charts)
                print("All ROC and SHAP charts saved!")
                #st.write(results_dictonary)
                path_name = os.path.join(algorithm_folder, f"{exp_name}_results.joblib")
                joblib.dump(results_dictonary, path_name)

                # generate the results table
                results_df = generate_results_table(results_dictonary)
                table_name = os.path.join(algorithm_folder, f"{exp_name}_results.xlsx")
                results_df.to_excel(table_name, index=False, engine='openpyxl')
                results_df.to_excel(table_name, index=False)
                results_df.to_excel(table_name, index=False)
                expand_cell_excel(table_name)
                wrap_text_excel(table_name)
                grid_excel(table_name)

                # Convert results_df to list of dictionaries
                results_dic = results_df.to_dict(orient='records')
                # get the current time
                current_datetime = datetime.now()
                current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                # get test dataset name
                test_name = uploaded_test_set.name if data_name_test == None else data_name_test
                try:
                    result = {
                        "exp_name": exp_name,
                        "type": model_type,
                        "test set": test_set_name,
                        "threshold used": threshold_type,
                        "results_dic": results_dictonary,
                        "results_table": results_dic,
                        'dataset used': test_name,
                        "input variables": input_variables,
                        "input variables (original)": input_cols_og,
                        'outcomes': selected_outcomes,
                        "time_created": current_time,
                    }

                    results.insert_one(result) # insert one dictonary
                except:
                    print("Results size is too large. Will save filepaths instead.")

                    result = {
                        "exp_name": exp_name,
                        "type": model_type,
                        "test set": test_set_name,
                        "threshold used": threshold_type,
                        "results_dic": path_name,
                        "results_table": results_dic,
                        'dataset used': test_name,
                        "input variables": input_variables,
                        "input variables (original)": input_cols_og,
                        'outcomes': selected_outcomes,
                        "time_created": current_time,
                    }
                    results.insert_one(result) # insert one dictonary

                print(f"Testing '{exp_name}' completed successfully!")
                update_status_when_done("Completed", "Training complete")
        else:
            print("Test Result of the same name already exists. Please change Test Set name.")
            #update_status("Failed", "Error Saving the test data.")

            jobs.update_one(
                {"job_id": job_id},
                {"$set": {
                    "status": "Failed",
                    "message": "Error Saving the test data.",
                    "last_updated": datetime.now(),
                    "expires_at": datetime.now() + timedelta(days=1)
                }}
            )
    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()

        print(f"Error Occured. Experiment '{exp_name}' failed!")
        traceback.print_exc()

        jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "Failed",
                "message": error_message,
                "error_trace": stack_trace,
                "last_updated": datetime.now(),
                "expires_at": datetime.now() + timedelta(days=1)
            }}
        )

    finally:
        stop_heartbeat = True
        heartbeat_thread.join(timeout=1)

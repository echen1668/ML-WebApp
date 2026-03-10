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

# import module
import streamlit as st
import os
import joblib
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
import threading
mpl_lock = threading.Lock()

#global exp_list # list of experiments
selected_outcomes = None

# --- Page Configuration ---
st.set_page_config(
    page_title="(Sklearn) Upload and Test ML Model",
    page_icon="🧪",
    layout="wide"
)

# Check if client_name was passed
cookies = EncryptedCookieManager(prefix="mlhub_", password="some_secret_key")
if not cookies.ready():
    st.stop()

# Check cookies first
#if "client_name" in cookies:
#   st.session_state["client_name"] = cookies["client_name"]
    #st.write(cookies["client_name"])
#else:
#    st.error("No found")

def get_client_name():
    try:
        if cookies.ready() and "client_name" in cookies:
            return cookies["client_name"]
    except Exception:
        pass
    return None

@st.cache_resource
def get_db(client_name):
    client = MongoClient(client_name, 27017, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client.machine_learning_database

st.session_state["client_name"] = get_client_name()

# then check in session state
if "client_name" not in st.session_state:
    st.error("No database connection found. Please go back to the main page.")
    st.stop()
# get the clinet
client_name = st.session_state["client_name"]

# connect to database
#client = MongoClient('10.14.1.12', 27017)
#client = MongoClient(client_name, 27017)


db = get_db(client_name)
#try:
    # create the database if it does not already exists
#    db = get_db(client_name)
#except Exception as e:
#    st.error("❌ Database connection failed")
#    st.exception(e)
#    st.stop()    

# create tables for models in the databse
models = db.models

# create the results if it does not already exists
jobs = db.jobs
db.jobs.create_index("expires_at", expireAfterSeconds=0)
cleanup_stale_jobs(db) # clean up an 'zombie jobs'

# create the results if it does not already exists
results = db.results

# create the results if it does not already exists
datasets = db.datasets

# get all testing data names from database
data_names_list_test = db.datasets.distinct("data_name", {"type": "Test"})

# get all unique exp. names from results collection
exp_names = db.models.distinct("exp_name", {"type": "Native"})


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
def save_testdata(exp_name, test_set, test_set_name, data_name_test, uploaded_test_set):
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

    # Specifying the list of outcomes with no results for demonstration purposes
    #print(f'Outcomes with no results are {no_results}')

    # Writing the output to a text file
    #with open("outcomes_no_results.txt", "w") as file:
        #file.write(f"Outcomes with no results are {no_results}")

    # Convert the list of rows into a DataFrame
    results_df = pd.DataFrame(rows)
    return results_df
    

def plot_feature_importance(feature_importance_dic, directory_name):
    outcomes = list(feature_importance_dic.keys())
    st.write(f'Outcomes {outcomes}')
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

        # plot the ROC Curves
        plot_roc(roc_payload['res'], roc_payload['res_array'], roc_payload['y_test'], roc_payload['probas_test'], roc_payload['algo_name'], roc_payload['outcome_name'], roc_payload['algorithm_folder'])

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
    
    # get connected to MongoDB database inside worker
    client = MongoClient(testing_setup["client_name"], 27017)
    db = client.machine_learning_database
    #db = get_db(testing_setup["client_name"])

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

    metric_dic = {} # dictionary to store all important metrics

    #with st.spinner(f"Uploading Model for {algo_name} on {o_name}..."):
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
    st.write("  Outcome Name: ", outcome_name)
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
        st.error("Input Columns do not match with data set.")
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

        #st.write('Testing Complete!')
    #with st.spinner(f"Testing on Data Done. Now calcuating ROC and SHAP values for {algo_name} on {outcome_name}..."):
    #st.write('Results Saving...')
    # plot the ROC Curves
    #plot_roc(res, res_array, y_test, probas_test, algo_name, outcome_name, algorithm_folder)

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
        st.write("Algorithm: ", algo_name)
        
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
                'client_name': client_name
            }

            testing_setups.append(testing_setup)

    #st.write(testing_setups)

    st.info("Testing Models have started...")
    with st.spinner("Testing Models..."):
        all_results = []
        t2 = time.time()
        #with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor: # test the models (use ProcessPoolExecutor to test as many models as possible to improve speed)
        #with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor: # test the models (use ThreadPoolExecutor to test as many models as possible to improve speed)
        #st.write(os.cpu_count())
        #    all_results = list(executor.map(test_model, testing_setups))

        all_results = Parallel( # test the models (use Parallel to make the process faster)
            n_jobs=max(1, multiprocessing.cpu_count() - 1),
            backend="loky", # IMPORTANT (default, but explicit is good)
            verbose=10
        )(
            delayed(test_model)(setup) for setup in testing_setups
        )

        t1 = time.time()

    st.success("Testing Model is Done!")
    st.write(f'The Time Parallel Computing Took :{t1-t2} seconds')
    #st.write(all_results)

    # save a textfile telling the time it took to test models
    algorithm_folder = os.path.join("Results", project_name, test_set_name)
    filename_time = os.path.join(algorithm_folder, "testing_time.txt")
    f = open(filename_time, "w", encoding="utf-8")
    f.write(f'The Time Parallel Computing Took :{t1-t2} seconds')
    f.close()

    return [r for r in all_results if isinstance(r, tuple)]
    

# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Testing_Models_Options.py")  # Redirect to the main back

# get all current jobs running and check their status
jobs_list = list(db.jobs.find({}, {"_id": 0}).sort("created_at", -1))
with st.expander("💼 Show All Job Status"):
    for job in jobs_list:
        st.markdown(f"### {job['exp_name']}")
        for key, value in job.items():
            st.write(f"**{key}**:", value)
        st.divider()

# Title
st.title("🧪 Upload and Test ML Model (Sklearn)")
st.write("Test multiple models based on the Sklearn/Native framework using new/unseen data.")

#st.write(data_names_list_test)

# --- Step 1: Retrive a ML Experiment ---
st.header("Step 1: Retrive a ML Experiment")

# Dropdown to select the ML model
exp_name = st.selectbox("Select the ML model(s)", exp_names, index=None, placeholder="Select One...")

#if exp_name == None:
#    test_set = None

# get the model data
exp_dic = models.find_one({"exp_name": exp_name})

# get the needed values
model_path = exp_dic['model_path'] if exp_dic is not None else None
configuration = exp_dic['configuration'] if exp_dic is not None else None
model_type = exp_dic['type'] if exp_dic is not None else None
num_algo = len(exp_dic['algorithms']) if exp_dic is not None else None
input_variables = exp_dic['input variables'] if exp_dic is not None else None
try:
    input_cols_og = exp_dic['input variables (original)'] if exp_dic is not None else None
except:
    input_cols_og = input_variables
outcomes = exp_dic['outcomes'] if exp_dic is not None else None
train_data = exp_dic['train_data'] if exp_dic is not None else None

if exp_dic != None:
    # get the model data and their configuration
    exper_list = list(configuration[list(configuration.keys())[0]].keys())[1:]
    #st.write(exp_list)
    # dropdown section to show excluded outcomes
    with st.expander("🚫 Show Excluded Outcomes"):
        # Dropdown to select an exp/algorthim
        exp_item = st.selectbox("Select an exp.", exper_list, help="Select a specfic exp. from the ML experiment.", placeholder="Select One...")

        algorithm = configuration[list(configuration.keys())[0]][exp_item]['algorithm']
        # uploaded textfile showing excluded columns
        file_path_excluded_labels = f'Models/{exp_name}/{exp_item}/excluded_label_cols_setup.txt'
        try:
            with open(file_path_excluded_labels, 'r') as file:
                content = file.read()  # Reads the entire content of the file
                #st.write(content)
                st.text_area(f"Excluded Outcomes for **{algorithm}**", content, height=300)
        except FileNotFoundError:
            st.error(f"Error: The file '{file_path_excluded_labels}' was not found.")
        except Exception as e:
            st.error(f"An error occurred while reading the file '{file_path_excluded_labels}': {e}")


# get the time created
try:
    time_created = exp_dic['time_created']
except:
    time_created = 'N/A'

# check if exp_name is Native if there is one
if exp_name == None:
    exp_dic = None
    model_path = None
    model_type = None
    input_variables = None
    test_set = None
elif exp_name != None and model_type != 'Native':
    st.write("Results is not Native.")
    exp_dic = None
    model_path = None
    model_type = None
    input_variables = None
    test_set = None


# give some information about the ML Experiment
if exp_dic != None:
    with st.expander("▶️ ML Experiment Info"):
        # write all model content in expander
        st.markdown(f'##### <u>{exp_name}</u>', unsafe_allow_html=True)
        st.write(f'**Model Type:** {model_type}')
        st.write(f'**Number of Algorithms:** {num_algo}')
        st.write(f'**Train Data:** {train_data}')
        st.write(f'**Time Created:** {time_created}')


# --- Step 2: Upload Data (Dynamic UI) ---
st.header("Step 2: Upload Data")

#user chooses whatever to upload the data or retrive a past data set from the database
data_options = st.radio("Choose an option:", ["Upload a testing set", "Retrive testing set from database"])

if data_options == "Upload a testing set":
    data_name_test = None
    dataset_uploader_key = "main_dataset_uploader"
    # File uploader for the test
    uploaded_test_set = st.file_uploader("Upload a Testing Data Set", type=['csv', 'xlsx'], key=dataset_uploader_key)

    # upload the test set
    if uploaded_test_set is not None:
        try:
            # Determine file type and read accordingly
            test_set = load_data(uploaded_test_set.name, uploaded_test_set)
            
            # Replace inf and -inf with NaN
            test_set = test_set.replace([np.inf, -np.inf], np.nan)
            
            # Display the DataFrame
            st.write("### Test Set:")
            st.dataframe(test_set)

            # check if upload test set has the require input and output variables
            test_cols = test_set.columns.to_list()
            # Check if  outcomes and input_cols_og is a subset of test_cols
            is_subset = all(x in test_cols for x in outcomes + input_cols_og)

            #is_subset = True
            if is_subset == False:
                st.error("Uploaded test set does not have the required variables.")

                # Convert to sets
                required_cols = set(outcomes + input_cols_og)
                test_cols_set = set(test_cols)
                missing_cols = required_cols - test_cols_set
                st.error(f'Missing Features: {missing_cols}')

        except Exception as e:
            st.error(f"Error loading file: {e}")
            is_subset = False
else:
    uploaded_test_set = None
    # Dropdown to select the testing dataset
    data_name_test = st.selectbox("Select a Testing Dataset from the database:", data_names_list_test, index=None, placeholder="Select One...")
    if data_name_test:
        try:
            # upload the testing set
            test_set = upload_data(os.path.join("Data Sets",data_name_test))
            # Replace inf and -inf with NaN
            test_set = test_set.replace([np.inf, -np.inf], np.nan)

            # Display the DataFrame
            st.write("### Test Set:")
            st.dataframe(test_set)

            # check if upload test set has the require input and output variables
            test_cols = test_set.columns.to_list()
            # Check if  outcomes is a subset of test_cols
            is_subset = all(x in test_cols for x in outcomes + input_cols_og)
            #is_subset = True
            if is_subset == False:
                st.error("Uploaded test set does not have the required variables.")

                # Convert to sets
                required_cols = set(outcomes + input_cols_og)
                test_cols_set = set(test_cols)
                missing_cols = required_cols - test_cols_set
                st.error(f'Missing Features: {missing_cols}')
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            is_subset = False
    else:
        is_subset = False


# --- Step 3: Configure Testing Pipeline ---
st.header("Step 3: Configure Testing Pipeline")

if model_path is not None:
    try:
        # Determine file type and read accordingly
        models_dic = joblib.load(model_path)
        algos = list(models_dic.keys())
        all_algos = ["Select All"] + algos

        outcomes = list(models_dic[list(models_dic)[0]].keys())
        all_outcomes = ["Select All"] + outcomes

        # Select multiple algorithims for ML model testing
        selected_algos = st.multiselect("Select algorithims to the test its model with", all_algos, [])

        if "Select All" in selected_algos:
            selected_algos = algos

        # Select multiple outcomes for ML model testing
        selected_outcomes = st.multiselect("Select outcomes to the test its model with", all_outcomes, [])

        if "Select All" in selected_outcomes:
            selected_outcomes = outcomes

        #st.write(selected_outcomes)
        #st.write(len(selected_outcomes))

        # Initialize session state variable
        if "show_values_models_dic" not in st.session_state:
            st.session_state.show_values_models_dic = False

        # Button to display values of outcome_dic
        if st.button('Display the Values'):
            st.session_state.show_values_models_dic = True  # Set state to show values

        # Button to hide values (appears only when values are shown)
        if st.session_state.show_values_models_dic:
            st.write(models_dic)
            if st.button('Hide the Values'):
                st.session_state.show_values_models_dic = False  # Reset state to hide values
                st.rerun()  # Refresh the page to update UI

    except Exception as e:
        st.error(f"Error loading file: {e}")

# Let user specify test set name
test_set_name = st.text_input("Enter Name of the Test Set", "test set")

# Dropdown to select the threshold to use to test the models
threshold_type = st.selectbox("Select a Threshold type", ['youden', 'mcc', 'ji', 'f1'])

# START BUTTON FOR USER
if model_path is not None and (uploaded_test_set or data_name_test) is not None and len(selected_outcomes)!=0 and is_subset==True:
    # --- Step 4: Execute ---
    st.header("Step 4: Begin Testing")

    # button to test the models
    if st.button('Test the models 🧪'):
        is_success = save_testdata(exp_name, test_set, test_set_name, data_name_test, uploaded_test_set) # save the testing data into database first
        
        if is_success == "Success":
            # call the function to test models
            results_list = test_models(models_dic, configuration, selected_algos, selected_outcomes, input_variables, test_set, exp_name, test_set_name, cutoff_index=threshold_type)
            if results_list == "Error": # if test_models function returns an error
                st.error(f"❌ Testing '{exp_name}' Failed. Go back and check testing configuration.")
            else:
                with st.spinner("Saving all results..."): # if no error, then save the results
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
                    st.success("All ROC and SHAP charts saved!")
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
                        st.info("Results size is too large. Will save filepaths instead.")

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

                st.success(f"✅ Testing '{exp_name}' completed successfully!")
                st.subheader("Jump to Visualizing Results") # redirect to the testing section
                st.page_link("pages/Visualize_Multi_Results (Native).py", label="Visualize Results", icon="📊")
        else:
            st.error("Test Result of the same name already exists. Please change Test Set name.")


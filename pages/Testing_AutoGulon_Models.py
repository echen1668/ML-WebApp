import time

import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
import csv
#import magic 
import pickle
import random 
from random import randint
from random import uniform
from pathlib import Path
import json
from scipy import stats
import os
import joblib as joblib
from joblib import dump, load
#np.random.seed(1000)
rstate = 12

selected_outcomes = None

# import module
import streamlit as st
import os
import joblib
from datetime import datetime
import pprint
import pymongo
from pymongo import MongoClient
from streamlit_cookies_manager import EncryptedCookieManager
from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel, upload_data, save_data
from roctools import full_roc_curve, plot_roc_curve

# --- Page Configuration ---
st.set_page_config(
    page_title="(AutoGulon) Upload and Test ML Model",
    page_icon="🧪",
    layout="wide"
)

# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Testing_Models_Options.py")  # Redirect to the main back

# Title
st.title("🧪 Upload and Test ML Model (AutoGulon)")
st.write("Test multiple models based on the AutoGulon framework using new/unseen data.")

# Check if client_name was passed
cookies = EncryptedCookieManager(prefix="mlhub_", password="some_secret_key")
if not cookies.ready():
    st.stop()

# Check cookies first
if "client_name" in cookies:
    st.session_state["client_name"] = cookies["client_name"]
    #st.write(cookies["client_name"])
#else:
    #st.error("No found")

# then check in session state
if "client_name" not in st.session_state:
    st.error("No database connection found. Please go back to the main page.")
    st.stop()

client_name = st.session_state["client_name"]

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

# get all unique exp. names from results collection
exp_names = db.models.distinct("exp_name", {"type": "AutoGulon"})

# get all training data names from database
data_names_train = db.datasets.distinct("data_name", {"type": "Train"})

# get all testing data names from database
data_names_list_test = db.datasets.distinct("data_name", {"type": "Test"})


def sanitize_filename(filename):
    """Remove or replace invalid characters from filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# testing models function
def test_model(models, test_data_raw, input_columns, outcomes, train_data_raw=None, cutoff_index='youden', algorithm_folder="Logfiles"):
    #st.write("Currently testing the models...")

    #placeholder = st.empty()  # Create a placeholder

    # Write the starting message
    #placeholder.write("Currently testing the models...")

    print(f"Cutoff: {cutoff_index}.")
    
    # results dictionary
    results_dictonary = {}
    
    # dictonary to store ground truths, predictions, and probablites from the test set
    outcome_dic = {}
    
    # dictonary for feature importance
    feature_importance_dic = {}

    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for logfile

    # create a log file for the outcome testing
    log_filename = os.path.join(algorithm_folder, "logfile.txt")
    f = open(log_filename, "w", encoding="utf-8")
    
    for outcome in outcomes:
        with st.spinner(f"Testing Model for {outcome}..."):

            f.write("_____________________________________________________________________________________________________")
            f.write("\nOutcome: %s"% outcome)

            # create the train/test set for the specifc input variables and specific outcome.
            if train_data_raw is not None:
                input_train = train_data_raw[input_columns]
                train_data = pd.concat([input_train, train_data_raw[[outcome]]], axis=1)
            
            input_test = test_data_raw[input_columns]
            test_data = pd.concat([input_test, test_data_raw[[outcome]]], axis=1)
            
            if train_data_raw is not None:
                y_train = train_data[outcome]
                print(f'Training Data: {train_data}')

            y_test = test_data[outcome]
            print(f'Training Data: {test_data}')

            f.write("\nTraining Data Length: %s"% len(test_data))
            
            # Count positives and negatives
            positives = np.sum(y_test == 1)  # Count instances of 1
            negatives = np.sum(y_test == 0)  # Count instances of 0
            
            print("Postive Count (on training set):", positives)
            print("Negative Count (on training set):", negatives)
            
            #if positives < 10:
                #continue

            #check_same_feature_set(train_data, test_data)
            print(test_data[outcome].value_counts())
                
            # create a states able for metric on the test set
            #res, res_array = full_roc_curve(y_test.to_numpy(), y_pred.to_numpy())
                
            try:
                # get the model
                model = models[outcome]['Model']
                print(f'Models Label: {model.label}')
                f.write("\nModel: %s"% model)
                #model = models
                    
                outcome_dic[outcome] = {}
                    
                #y_pred = model.predict(test_data.drop(columns=[outcome])) # get the predictions for test set
                y_proba = model.predict_proba(test_data.drop(columns=[outcome])) # Prediction Probabilities for test set
                evaluation = model.evaluate(test_data, silent=True)
                    
                # save ground truths, predictions, and probabilities
                outcome_dic[outcome]['Ground Truths'] = y_test
                #outcome_dic[outcome]['Predictions'] = y_pred
                outcome_dic[outcome]['Probability Scores'] = y_proba
            
                if train_data_raw is not None:
                    # create a states able for metric on the train set
                    #y_pred_train = model.predict(train_data.drop(columns=[outcome]))
                    y_proba_train = model.predict_proba(train_data.drop(columns=[outcome]))
                    res_train, res_array_train = full_roc_curve(y_train.to_numpy(), y_proba_train[1].to_numpy(), index=cutoff_index)
                    evaluation['AUROC Score (Train)'] = res_train['auc']
                    evaluation['AUROC CI Low (Train)'] = res_train['auc_cilow']
                    evaluation['AUROC CI High (Train)'] = res_train['auc_cihigh']
                    evaluation['P (Train)'] = res_train['P'].astype(float)
                    evaluation['N (Train)'] = res_train['N'].astype(float)
                    evaluation['TP (Train)'] = res_train['TP']
                    evaluation['FP (Train)'] = res_train['FP']
                    evaluation['TN (Train)'] = res_train['TN']
                    evaluation['FN (Train)'] = res_train['FN']
                    
                # create a states able for metric on the test set
                res, res_array = full_roc_curve(y_test.to_numpy(), y_proba[1].to_numpy(), index=cutoff_index)
                print("Results Array: ", res)
                f.write("\nResults Array: %s"% res)
                evaluation['TPR'] = res['tpr']
                evaluation['TNR'] = res['tnr']
                evaluation['FPR'] = res['fpr']
                evaluation['FNR'] = res['fnr']
                evaluation['PPV'] = res['ppv']
                evaluation['NPV'] = res['npv']
                    
                evaluation['AUROC Score'] = res['auc']
                evaluation['AUROC CI Low'] = res['auc_cilow']
                evaluation['AUROC CI High'] = res['auc_cihigh']
                evaluation['cutoff type'] = cutoff_index
                evaluation['cutoff'] = res['cutoff_mcc'] if cutoff_index=='mcc' else (res['cutoff_ji'] if cutoff_index=='ji' else (res['cutoff_f1'] if cutoff_index=='f1' else res['cutoff_youden']))
                    
                #st.write("Cutoff Index: ", evaluation['cutoff'])

                y_pred = [1 if p >= evaluation['cutoff'] else 0 for p in y_proba[1]] # predict with test set
                test_acc = accuracy_score(y_test, y_pred) # test accuracy
                print("Test Accuracy:", (test_acc * 100))
                evaluation['Test Accuracy'] = test_acc
                    
                # Extract TP, FP, TN, FN
                evaluation['TP'] = res['TP']
                evaluation['FP'] = res['FP']
                evaluation['TN'] = res['TN']
                evaluation['FN'] = res['FN']
                    
                evaluation['P'] = res['P'].astype(float)
                evaluation['N'] = res['N'].astype(float)
                    
                evaluation['precision'] = res['precision']
                evaluation['recall'] = res['recall']
                evaluation['f1 score'] = res['f1 score']
                    
                evaluation['Ground Truths'] = y_test.tolist()
                evaluation['Predictions'] = y_pred
                evaluation['Probability Scores'] = y_proba.values.tolist()
                    
                print(evaluation)
            except:
                st.error(f"Unable to make prediction for {outcome}.")
                #print(f'Its value count is {y_test.value_counts()}')
                continue

        with st.spinner(f"Model Testing for {outcome} is Done. Now getting the Feature Importance..."):    
            print("Feature Importance....")
            f.write("Feature Importance....")
            original_features = model.features(feature_stage='original')
            #st.write(f'Aviable features: {original_features}')
            # Get feature importance
            feature_importance_df = model.feature_importance(test_data, features=original_features, time_limit=500)
            feature_importance_df = feature_importance_df.reset_index(names='feature')
            print("Feature Importance Array: ",feature_importance_df)

            f.write("\nFeature Importance Array: %s"% feature_importance_df)

            # Sort by importance
            feature_importance_df_sorted = feature_importance_df.sort_values('importance', ascending=False)
            print(feature_importance_df_sorted)
            #st.dataframe(feature_importance_df_sorted)
            feature_importance_dic[outcome] = feature_importance_df_sorted

            # Get top 10 most important feature names
            top_features = feature_importance_df_sorted['feature'].head(10).tolist()
            #all_features = feature_importance_df_sorted['feature'].tolist()
            #st.write(f"Top 10 important features: {top_features}")
            f.write("\nTop 10 important features: %s" % top_features)

            #except:
            #    st.error(f"Unable to make the feature importance table for {outcome}.")
            
            results_dictonary[outcome] = {}
            model.leaderboard(test_data)
            print(f'Best Model for {outcome} is {model.model_best}.')
            results_dictonary[outcome]['best_model'] = model.model_best
            results_dictonary[outcome]['evaluation'] = evaluation
            results_dictonary[outcome]['leaderboard'] = model.leaderboard(test_data).to_dict(orient='records')
            try:
                results_dictonary[outcome]['feature importance'] = feature_importance_df_sorted.to_dict(orient='records')
                #results_dictonary[outcome]['top features'] = all_features
            except:
                st.error(f"Unable to make the feature importance table for {outcome}.")

            st.success(f'Testing for {outcome} is complete.')
            f.write(f'Testing for {outcome} is complete.')
            print("_________________________________________________________")
    
    #st.write("Model Testing is Complete!") 
    # Write the ending message
    #placeholder.empty()  # Clears the output
    f.write("Model Testing is Complete!")
    st.success("Model Testing is Complete!")

    f.close()
    
    return results_dictonary, outcome_dic, feature_importance_dic

def generate_results_table(results_dictonary, outcomes):
    # Initialize an empty list to collect rows
    rows = []
    no_results = []

    for outcome in outcomes:
        print(outcome)
        try:
            print(results_dictonary[outcome])
        except:
            print(f'No results exists for {outcome}.')
            no_results.append(outcome)
            continue
            
        leaderboard = results_dictonary[outcome]['leaderboard']
        df_leaderboard = pd.DataFrame(leaderboard)
        best_model = df_leaderboard.loc[0]['model']
        best_model_auroc = df_leaderboard.loc[0]['score_test']

        if 'AUROC Score (Train)' in list(results_dictonary[outcome]['evaluation'].keys()):
            new_row = {'Outcome': outcome,
                  'AUROC Score': results_dictonary[outcome]['evaluation']['AUROC Score'],
                  'AUROC CI Lower': results_dictonary[outcome]['evaluation']['AUROC CI Low'],
                  'AUROC CI Upper': results_dictonary[outcome]['evaluation']['AUROC CI High'],
                  'Accuracy': results_dictonary[outcome]['evaluation']['Test Accuracy'],
                  'Precision': results_dictonary[outcome]['evaluation']['precision'],
                  'Recall': results_dictonary[outcome]['evaluation']['recall'],
                  'F1 Score': results_dictonary[outcome]['evaluation']['f1 score'],
                  'TPR': results_dictonary[outcome]['evaluation']['TPR'], # same as Sensitivity 
                  'TNR': results_dictonary[outcome]['evaluation']['TNR'], # same as Specificity 
                  'FPR': results_dictonary[outcome]['evaluation']['FPR'], 
                  'FNR': results_dictonary[outcome]['evaluation']['FNR'],
                  'PPV': results_dictonary[outcome]['evaluation']['PPV'],
                  'NPV': results_dictonary[outcome]['evaluation']['NPV'],
                  'TP': results_dictonary[outcome]['evaluation']['TP'],
                  'FP': results_dictonary[outcome]['evaluation']['FP'],
                  'TN': results_dictonary[outcome]['evaluation']['TN'],
                  'FN': results_dictonary[outcome]['evaluation']['FN'],
                  'Cutoff value': results_dictonary[outcome]['evaluation']['cutoff'],
                  'Best Model': best_model,
                  'Best Model AUROC Score': best_model_auroc,
                  'P': results_dictonary[outcome]['evaluation']['P'],
                  'N': results_dictonary[outcome]['evaluation']['N'],
                  'AUROC Score (Train)': results_dictonary[outcome]['evaluation']['AUROC Score (Train)'],
                  'AUROC CI Lower (Train)': results_dictonary[outcome]['evaluation']['AUROC CI Low (Train)'],
                  'AUROC CI Upper (Train)': results_dictonary[outcome]['evaluation']['AUROC CI High (Train)'],
                  'P (Train)': results_dictonary[outcome]['evaluation']['P (Train)'],
                  'N (Train)': results_dictonary[outcome]['evaluation']['N (Train)']}
        else:
            new_row = {'Outcome': outcome,
                  'AUROC Score': results_dictonary[outcome]['evaluation']['AUROC Score'],
                  'AUROC CI Lower': results_dictonary[outcome]['evaluation']['AUROC CI Low'],
                  'AUROC CI Upper': results_dictonary[outcome]['evaluation']['AUROC CI High'],
                  'Accuracy': results_dictonary[outcome]['evaluation']['Test Accuracy'],
                  'Precision': results_dictonary[outcome]['evaluation']['precision'],
                  'Recall': results_dictonary[outcome]['evaluation']['recall'],
                  'F1 Score': results_dictonary[outcome]['evaluation']['f1 score'],
                  'TPR': results_dictonary[outcome]['evaluation']['TPR'], # same as Sensitivity 
                  'TNR': results_dictonary[outcome]['evaluation']['TNR'], # same as Specificity 
                  'FPR': results_dictonary[outcome]['evaluation']['FPR'], 
                  'FNR': results_dictonary[outcome]['evaluation']['FNR'],
                  'PPV': results_dictonary[outcome]['evaluation']['PPV'],
                  'NPV': results_dictonary[outcome]['evaluation']['NPV'],
                  'TP': results_dictonary[outcome]['evaluation']['TP'],
                  'FP': results_dictonary[outcome]['evaluation']['FP'],
                  'TN': results_dictonary[outcome]['evaluation']['TN'],
                  'FN': results_dictonary[outcome]['evaluation']['FN'],
                  'Cutoff value': results_dictonary[outcome]['evaluation']['cutoff'],
                  'Best Model': best_model,
                  'Best Model AUROC Score': best_model_auroc,
                  'P': results_dictonary[outcome]['evaluation']['P'],
                  'N': results_dictonary[outcome]['evaluation']['N']}
            
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
    print(f'Outcomes with no results are {no_results}')

    # Writing the output to a text file
    with open("outcomes_no_results.txt", "w") as file:
        file.write(f"Outcomes with no results are {no_results}")
            
    # Convert the list of rows into a DataFrame
    results_df = pd.DataFrame(rows)
    return results_df

def plot_feature_importance(feature_importance_dic, directory_name):
    outcomes = list(feature_importance_dic.keys())
    #st.write(f'Outcomes {outcomes}')
    for outcome in outcomes:
        st.write(f'Outcome {outcome}')
        # get the table
        df_feature_importance = feature_importance_dic[outcome]
        st.write(df_feature_importance)
        
        # the abs value of feature importance values
        df_feature_importance['importance (abs)'] = df_feature_importance['importance'].abs()
        
        # Sort the DataFrame by the column values
        #df_feature_importance = df_feature_importance.sort_values(by='importance (abs)', ascending=False)

        # get the names of the top 10 most important features
        top_features = df_feature_importance['feature'].head(10).tolist()
        top_importances = df_feature_importance['importance (abs)'].head(10).tolist()
        
        # plot the 10 top most important features
        fig, ax = plt.subplots(figsize=(6, 6))
        #ax = df_feature_importance['importance (abs)'][:10].plot(kind='bar')
        ax.bar(top_features, top_importances, color='skyblue')
        ax.set_title(f'Feature Importance for {outcome} on {exp_name} (Top 10)', fontsize=8)
        ax.set_ylabel('Importance (abs)', fontsize=7)
        ax.set_xlabel('Feature', fontsize=7)

        # Add text labels
        for i, v in enumerate(top_importances):
            ax.text(i, v + 0.0005, f'{v:.3f}', ha='center')

        # Set y-axis range from 0 to 1
        ax.set_ylim(0, max(top_importances) + 0.01)

        # Adjust x-axis labels for readability
        plt.xticks(top_features, rotation=45, ha='right', fontsize=8)  # Rotate and align right
        plt.yticks(fontsize=8)
        
        # Save the plot as a PNG file
        os.makedirs(directory_name, exist_ok=True)
        file_name = f'{directory_name}/{sanitize_filename(outcome)}_feature_importance.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

        plt.close()
        

        #plt.show()

# --- Step 1: Retrive a ML Experiment ---
st.header("Step 1: Retrive a ML Experiment")

# Dropdown to select the ML model
exp_name = st.selectbox("Select the ML model(s)", exp_names, index=None, placeholder="Select One...")

# get the model data
exp_dic = models.find_one({"exp_name": exp_name})

# get the model data and their configuration
if exp_dic != None:
    # dropdown section to show excluded outcomes
    with st.expander("🚫 Show Excluded Outcomes"):
        # uploaded textfile showing excluded columns
        file_path_excluded_labels = f'Models/{exp_name}/excluded_label_cols_setup.txt'
        try:
            with open(file_path_excluded_labels, 'r') as file:
                content = file.read()  # Reads the entire content of the file
                #st.write(content)
                st.text_area(f"Excluded Outcomes", content, height=300)
        except FileNotFoundError:
            st.error(f"Error: The file '{file_path_excluded_labels}' was not found.")
        except Exception as e:
            st.error(f"An error occurred while reading the file '{file_path_excluded_labels}': {e}")

# get the needed values
model_path = exp_dic['model_path'] if exp_dic is not None else None
model_type = exp_dic['type'] if exp_dic is not None else None
input_variables = exp_dic['input variables'] if exp_dic is not None else None
outcomes = exp_dic['outcomes'] if exp_dic is not None else None
train_data = exp_dic['train_data'] if exp_dic is not None else None

# get the time created
try:
    time_created = exp_dic['time_created']
except:
    time_created = 'N/A'

# check if exp_name is AutoGulon if there is one
if exp_name == None:
    exp_dic = None
    model_path = None
    model_type = None
    input_variables = None
    test_set = None
elif exp_name != None and model_type != 'AutoGulon':
    st.error("Results is not AutoGulon")
    exp_dic = None
    model_path = None
    model_type = None
    input_variables = None

# give some information about the ML Experiment
if exp_dic != None:
    with st.expander("▶️ ML Experiment Info"):
        # write all model content in expander
        st.markdown(f'##### <u>{exp_name}</u>', unsafe_allow_html=True)

        st.write(f'**Model Type:** {model_type}')
        st.write(f'**Train Data:** {train_data}')
        st.write(f'**Time Created:** {time_created}')


# --- Step 2: Upload Data (Dynamic UI) ---
st.header("Step 2: Upload Data")

#user chooses whatever to upload the data or retrive a past data set from the database
data_options = st.radio("Choose an option:", ["Upload a testing set", "Retrive testing set from database"])

if data_options == "Upload a testing set":
    data_name_test = None
    data_name_train = None
    # File uploader for the test
    uploaded_test_set = st.file_uploader("Upload a Testing Data Set")

    # File uploader for the train
    uploaded_train_set = st.file_uploader("(Optional) Upload the Training Data Set")

    # upload the test set
    if uploaded_test_set is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_test_set.name.endswith(".csv"):
                test_set = pd.read_csv(uploaded_test_set)
            else:
                test_set = pd.read_excel(uploaded_test_set)
            
            # Replace inf and -inf with NaN
            test_set = test_set.replace([np.inf, -np.inf], np.nan)
            
            # Display the DataFrame
            st.write("### Test Set:")
            st.dataframe(test_set)

            # check if upload test set has the require input and output variables
            test_cols = test_set.columns.to_list()
            # Check if  outcomes is a subset of test_cols
            is_subset = all(x in test_cols for x in input_variables + outcomes)
            if is_subset == False:
                st.error("Uploaded test set does not have the required variables.")
            else:
                is_subset = True

        except Exception as e:
            st.error(f"Error loading file: {e}")
            is_subset = False

    train_set = None
    # upload the train set
    if uploaded_train_set is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_train_set.name.endswith(".csv"):
                train_set = pd.read_csv(uploaded_train_set)
            else:
                train_set = pd.read_excel(uploaded_train_set)

            # Replace inf and -inf with NaN
            train_set = train_set.replace([np.inf, -np.inf], np.nan)
            
            # Display the DataFrame
            st.write("### Train Set:")
            st.dataframe(train_set)

            # check if upload train set has the require input and output variables
            train_cols = train_set.columns.to_list()
            # Check if  outcomes is a subset of test_cols
            is_subset = all(x in train_cols for x in input_variables + outcomes)
            if is_subset == False:
                st.error("Uploaded train set does not have the required variables.")
            else:
                is_subset = True

        except Exception as e:
            st.error(f"Error loading file: {e}")
            is_subset = False
else:
    uploaded_test_set = None
    uploaded_train_set = None

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
            # Check if outcomes is a subset of test_cols
            is_subset = all(x in test_cols for x in input_variables + outcomes)
            if is_subset == False:
                st.error("Uploaded test set does not have the required variables.")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            is_subset = False
    else:
        test_set = None
        is_subset = False

    # Dropdown to select the training dataset
    data_name_train = st.selectbox("(Optional) Select a Training Dataset from the database:", data_names_train, index=None, placeholder="Select One...")
    if data_name_train:
        # upload the testing set
        train_set = upload_data(os.path.join("Data Sets",data_name_train))
        # Replace inf and -inf with NaN
        train_set = train_set.replace([np.inf, -np.inf], np.nan)

        # check if upload train set has the require input and output variables
        train_cols = train_set.columns.to_list()
        # Check if  outcomes is a subset of test_cols
        is_subset = all(x in train_cols for x in input_variables + outcomes)
        if is_subset == False:
            st.error("Uploaded train set does not have the required variables.")
        else:
            is_subset = True
    else:
        train_set = None

# --- Step 3: Configure Testing Pipeline ---
st.header("Step 3: Configure Testing Pipeline")

if model_path is not None:
    try:
        # Determine file type and read accordingly
        models_dic = joblib.load(model_path)
        #outcomes = list(models_dic.keys())
        all_options = ["Select All"] + outcomes

        # Select multiple outcomes for ML model testing
        selected_outcomes = st.multiselect("Select outcomes to the test its model with", all_options, [])

        if "Select All" in selected_outcomes:
            selected_outcomes = outcomes

        st.write(selected_outcomes)
        st.write(len(selected_outcomes))

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

if input_variables is not None and model_path is not None and (uploaded_test_set or data_name_test) is not None and len(selected_outcomes)!=0 and is_subset==True:
    # --- Step 4: Execute ---
    st.header("Step 4: Begin Testing")

    # button to test the models
    if st.button('Test the models 🧪'):

        if db.results.find_one({"exp_name": exp_name, "test set": test_set_name}) == None:

            #save the test set if it hasn't already
            if data_name_test == None and uploaded_test_set.name not in data_names_list_test:
                st.info(f"Testing Dataset is saving in the database", icon="ℹ️")
                # create a list of ML exp.'s that the dataset was used on
                exp_list = [exp_name]
                #get the current time
                current_datetime = datetime.now()
                current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                # save test set in data folder and database
                os.makedirs("Data Sets", exist_ok=True)
                save_data(uploaded_test_set.name, test_set, os.path.join("Data Sets", uploaded_test_set.name))
                dataset_test = {
                        "data_name": uploaded_test_set.name,
                        "type": "Test",
                        "time_saved": current_time,
                        "data_path": os.path.join("Data Sets", uploaded_test_set.name),
                        "exps used": exp_list
                }
                datasets.insert_one(dataset_test)
            else:
                st.info(f"Testing Dataset of the same name is already in the database. Will be overwritten in the database", icon="ℹ️")
                test_name = uploaded_test_set.name if data_name_test == None else data_name_test
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

            algorithm_folder = os.path.join("Results", exp_name, test_set_name)
            os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results

            # call the function to test models
            results_dictonary, outcome_dic, feature_importance_dic = test_model(models_dic, test_set, input_variables, selected_outcomes, train_data_raw=train_set, cutoff_index=threshold_type, algorithm_folder=algorithm_folder)

            #st.write(results_dictonary)
            #st.write(feature_importance_dic)

            path_name = f"{algorithm_folder}/{exp_name}_results.joblib"
            joblib.dump(results_dictonary, path_name)

            # generate the feature importance charts
            chart_name = f"{algorithm_folder}/Feature Importance"
            plot_feature_importance(feature_importance_dic, chart_name)

            # generate the results table
            results_df = generate_results_table(results_dictonary, selected_outcomes)
            table_name = f"{algorithm_folder}/{exp_name}_results.xlsx"

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
            #st.write(results_dictonary)
            try:
                result = {
                    "exp_name": exp_name,
                    "type": model_type,
                    "test set": test_set_name,
                    "threshold used": threshold_type,
                    "results_dic": results_dictonary,
                    "results_table": results_dic,
                    'dataset used': test_name,
                    "time_created": current_time
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
                    "time_created": current_time
                }
                results.insert_one(result) # insert one dictonary

            st.success(f"✅ Testing '{exp_name}' completed successfully!")
            st.subheader("Jump to Visualizing Results") # redirect to the testing section
            st.page_link("pages/Visualize_Multi_Results (AutoGluon).py", label="Visualize Results", icon="📊")
        else:
            st.error("Test Result of the same name already exists. Please change Test Set name.")
import time

import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
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
import datetime
import pprint
import pymongo
from pymongo import MongoClient

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

# connect to database
client = MongoClient('10.14.1.12', 27017)

# create the database if it does not already exists
db = client.machine_learning_database

# create tables for models in the databse
models = db.models

# create the results if it does not already exists
results = db.results

# get all unique exp. names from results collection
exp_names = db.models.distinct("exp_name", {"type": "AutoGluon", "is_FS": False})


def sanitize_filename(filename):
    """Remove or replace invalid characters from filenames."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

# testing models function
def test_model(models, test_data_raw, input_columns, outcomes, train_data_raw=None, cutoff_index='youden'):
    #st.write("Currently testing the models...")

    placeholder = st.empty()  # Create a placeholder

    # Write the starting message
    placeholder.write("Currently testing the models...")

    print(f"Cutoff: {cutoff_index}.")
    
    # results dictionary
    results_dictonary = {}
    
    # dictonary to store ground truths, predictions, and probablites from the test set
    outcome_dic = {}
    
    # dictonary for feature importance
    feature_importance_dic = {}
    
    for outcome in outcomes:
        print(f'Working with outcome: {outcome}')

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
            #model = models
            
            outcome_dic[outcome] = {}
            
            y_pred = model.predict(test_data.drop(columns=[outcome])) # get the predictions for test set
            y_proba = model.predict_proba(test_data.drop(columns=[outcome])) # Prediction Probabilities for test set
            evaluation = model.evaluate(test_data, silent=True)
            
            # save ground truths, predictions, and probabilities
            outcome_dic[outcome]['Ground Truths'] = y_test
            outcome_dic[outcome]['Predictions'] = y_pred
            outcome_dic[outcome]['Probability Scores'] = y_proba
    
            if train_data_raw is not None:
                # create a states able for metric on the train set
                y_pred_train = model.predict(train_data.drop(columns=[outcome]))
                y_proba_train = model.predict_proba(train_data.drop(columns=[outcome]))
                res_train, res_array_train = full_roc_curve(y_train.to_numpy(), y_proba_train[1].to_numpy(), index=cutoff_index)
                evaluation['AUROC Score (Train)'] = res_train['auc']
                evaluation['AUROC CI Low (Train)'] = res_train['auc_cilow']
                evaluation['AUROC CI High (Train)'] = res_train['auc_cihigh']
                evaluation['P (Train)'] = res_train['P']
                evaluation['N (Train)'] = res_train['N']
            
            # create a states able for metric on the test set
            res, res_array = full_roc_curve(y_test.to_numpy(), y_proba[1].to_numpy(), index=cutoff_index)
            print("Results Array: ", res)
            evaluation['TPR'] = res['tpr']
            evaluation['TNR'] = res['tnr']
            evaluation['PPV'] = res['ppv']
            evaluation['NPV'] = res['npv']
            
            evaluation['AUROC Score'] = res['auc']
            evaluation['AUROC CI Low'] = res['auc_cilow']
            evaluation['AUROC CI High'] = res['auc_cihigh']
            evaluation['cutoff type'] = cutoff_index
            evaluation['cutoff'] = res['cutoff_mcc'] if cutoff_index=='mcc' else (res['cutoff_ji'] if cutoff_index=='ji' else (res['cutoff_f1'] if cutoff_index=='f1' else res['cutoff_youden']))
            
            print("Cutoff Index: ", evaluation['cutoff'])
            
            # Extract TP, FP, TN, FN
            evaluation['TP'] = res['TP']
            evaluation['FP'] = res['FP']
            evaluation['TN'] = res['TN']
            evaluation['FN'] = res['FN']
            
            evaluation['P'] = res['P']
            evaluation['N'] = res['N']
            
            evaluation['precision'] = res['precision']
            evaluation['recall'] = res['recall']
            
            evaluation['Ground Truths'] = y_test.tolist()
            evaluation['Predictions'] = y_pred.tolist()
            evaluation['Probability Scores'] = y_proba.values.tolist()
            
            print(evaluation)
        except:
            print(f"Unable to make prediction for {outcome}.")
            print(f'Its value count is {y_test.value_counts()}')
            continue
        
        try:
            print("Feature Importance....")
            # Get feature importance
            feature_importance_df = model.feature_importance(test_data, time_limit=500)

            print("Feature Importance Array: ",feature_importance_df)

            # Sort by importance
            feature_importance_df_sorted = feature_importance_df.sort_values('importance', ascending=False)
            print(feature_importance_df_sorted)
            feature_importance_dic[outcome] = feature_importance_df_sorted
        except:
            print(f"Unable to make the feature importance table for {outcome}.")
        
        results_dictonary[outcome] = {}
        model.leaderboard(test_data)
        print(f'Best Model for {outcome} is {model.model_best}.')
        results_dictonary[outcome]['best_model'] = model.model_best
        results_dictonary[outcome]['evaluation'] = evaluation
        results_dictonary[outcome]['leaderboard'] = model.leaderboard(test_data).to_dict(orient='records')
        try:
            results_dictonary[outcome]['feature importance'] = feature_importance_df_sorted.to_dict(orient='records')
        except:
            print("No feature importance dictonary exists.")

        print("_________________________________________________________")
    
    #st.write("Model Testing is Complete!") 
    # Write the ending message
    placeholder.empty()  # Clears the output
    placeholder.write("Model Testing is Complete!")
    
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
                  'Accuracy': results_dictonary[outcome]['evaluation']['accuracy'],
                  'Precision': results_dictonary[outcome]['evaluation']['precision'],
                  'Recall': results_dictonary[outcome]['evaluation']['recall'],
                  'TPR': results_dictonary[outcome]['evaluation']['TPR'], # same as Sensitivity 
                  'TNR': results_dictonary[outcome]['evaluation']['TNR'], # same as Specificity 
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
                  'Accuracy': results_dictonary[outcome]['evaluation']['accuracy'],
                  'Precision': results_dictonary[outcome]['evaluation']['precision'],
                  'Recall': results_dictonary[outcome]['evaluation']['recall'],
                  'TPR': results_dictonary[outcome]['evaluation']['TPR'], # same as Sensitivity 
                  'TNR': results_dictonary[outcome]['evaluation']['TNR'], # same as Specificity 
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

# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Testing_Models.py")  # Redirect to the main back

# Title
st.title("Upload and Test ML Model")

st.write("Test multiple models.")

# Dropdown to select the ML model
exp_name = st.selectbox("Select the ML model(s)", exp_names)

# get the model data
exp_dic = models.find_one({"exp_name": exp_name})

# get the needed values
model_path = exp_dic['model_path'] if exp_dic is not None else None
model_type = exp_dic['type'] if exp_dic is not None else None
input_variables = exp_dic['input variables'] if exp_dic is not None else None

# check if results_dict is AutoGulon
if model_type != 'AutoGluon' or exp_dic['is_FS'] == True:
    st.write("Results is not AutoGluon and/or has Feature Selection")
    exp_dic = None
    model_path = None
    model_type = None
    input_variables = None

# File uploader for the test
uploaded_test_set= st.file_uploader("Upload a Testing Data Set")

# File uploader for the train
uploaded_train_set= st.file_uploader("(Optional) Upload a Training Data Set")

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

    except Exception as e:
        st.error(f"Error loading file: {e}")

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

    except Exception as e:
        st.error(f"Error loading file: {e}")

if model_path is not None:
    try:
        # Determine file type and read accordingly
        models_dic = joblib.load(model_path)
        outcomes = list(models_dic.keys())
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

# Let user specify a directory name
directory_name = st.text_input("Enter Directory name:", "results")
os.makedirs(directory_name, exist_ok=True)

# Let user specify file name
file_name = st.text_input("Enter File Name", "results")

if input_variables is not None and model_path is not None and uploaded_test_set is not None and len(selected_outcomes)!=0:
    # button to test the models
    if st.button('Test the models'):

        try:
            # call the function to test models
            results_dictonary, outcome_dic, feature_importance_dic = test_model(models_dic, test_set, input_variables, selected_outcomes, train_data_raw=train_set, cutoff_index=threshold_type)

            st.write(results_dictonary)
            st.write(feature_importance_dic)

            path_name = f"{directory_name}/{file_name}.joblib"
            joblib.dump(results_dictonary, path_name)

            # generate the feature importance charts
            chart_name = f"{directory_name}/Feature Importance"
            plot_feature_importance(feature_importance_dic, chart_name)

            # generate the results table
            results_df = generate_results_table(results_dictonary, selected_outcomes)
            table_name = f"{directory_name}/{file_name}.xlsx"

            results_df.to_excel(table_name, index=False, engine='openpyxl')

            results_df.to_excel(table_name, index=False)
            results_df.to_excel(table_name, index=False)
            expand_cell_excel(table_name)
            wrap_text_excel(table_name)
            grid_excel(table_name)

            # Convert results_df to list of dictionaries
            results_dic = results_df.to_dict(orient='records')
            results_dic

            result = {
                "exp_name": exp_name,
                "type": model_type,
                "test set": test_set_name,
                "results_dic": results_dictonary,
                "results_table": results_dic,
                "results_path": f"{file_name}"
            }

            results.insert_one(result) # insert one dictonary
        except:
            st.write("Error has occured in testing. Check if feature set in data contains the required input variables from the model.")

if os.path.isfile(f"{directory_name}/{file_name}.xlsx") or os.path.isfile(f"{directory_name}/{file_name}.joblib"):
    # swithces to Visualize_Results
    if st.button('Visualize ML Results'):
        st.switch_page("pages/Visualize_Options.py")  # Redirect to visualize_results.py
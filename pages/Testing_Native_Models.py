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

selected_outcomes = None

# import module
import streamlit as st
import os
import joblib
from datetime import datetime
import pprint
import pymongo
from pymongo import MongoClient

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel, split, upload_data, save_data
from roctools import full_roc_curve, plot_roc_curve

# connect to database
client = MongoClient('10.14.1.12', 27017)

# create the database if it does not already exists
db = client.machine_learning_database

# create tables for models in the databse
models = db.models

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
        if key == "exp_type":
            continue
        
        if main_dic[key]["algorithm"] == algorithm:
            return main_dic[key]["options"]

def preprocessdata(df, numeric_cols, cutMissingRows='True', threshold=0.75, inf='replace with null', outliers='None', N=20000):
    if cutMissingRows == 'True':
        print("cutMissingRows")
        # Drop rows with missing values
        # computing number of columns
        cols = len(df.axes[1])
        print("Cuttoff", int(threshold * cols))
        df = df.dropna(thresh=int(threshold * cols))

    if inf == 'replace with null':
        print("replace with null")
        # Replace all inf values with null
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    elif inf == 'replace with zero':
        print("replace with zero")
        # Replace all inf values with null
        df.replace([np.inf, -np.inf], 0, inplace=True)

    if outliers == 'remove rows':
        print("remove rows")
        # Remove rows that have a value greater than N for any column. Default N is 20000
        for column  in df[numeric_cols]:
            df = df.drop(df.index[df[column] > N])
    elif outliers == 'log':
        print("log")
        # Log values that are greater than N for any column. Default N is 20000
        df[numeric_cols].apply(lambda x: np.where(x > N, np.log(x), x))

    return df

# testing models function
def test_models(model_dic, configuration, all_algorithms, all_outcomes, input_columns, test_data_raw, project_name, test_set_name, cutoff_index='youden'):
    results_dictonary = {}
    
    for algo_name in all_algorithms:
        st.write("Algorithm: ", algo_name)
        
        results_dictonary[algo_name] = {}
        
        for o_name in all_outcomes:
            outcome_dic = model_dic[algo_name][o_name]
            options = find_option_dic(configuration, project_name, algo_name)
            #st.write(options)
            algorithm_folder = os.path.join("Results", project_name, test_set_name, f'{algo_name} (results)', o_name)
            os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results
                
            metric_dic = {} # dictionary containg all important metrics

            with st.spinner(f"Uploading Model for {algo_name} on {o_name}..."):

                print(o_name)
                keys_list = list(outcome_dic.keys())
                #st.write(keys_list)
                model = outcome_dic['Model']
                features = outcome_dic['Features']
                outcome_name = outcome_dic['Outcome Name']
                df_res_train = outcome_dic['Train Set Res']
                df_array_res_train = outcome_dic['Train Set Res Array']
                print("  Model: ", model)
                print("  Features: ", features)
                st.write("  Outcome Name: ", outcome_name)
                print("  Res Train Table: ", df_res_train)
                print()
                if model == 'N/A' :
                    continue
                
                #print(input_columns)
                
                numeric_columns = outcome_dic['Numeric Columns']
                categorical_columns = outcome_dic['Categorical Columns']

                # get a copy of the dataset
                test_data = test_data_raw.copy()

                # Encoder
                if 'Encoder' in keys_list:
                    print("Encoder")
                    encoder = outcome_dic['Encoder']
                    encoded_cols = outcome_dic['Encoded Columns']
                    test_data[encoded_cols] = encoder.transform(test_data[categorical_columns])

                # preprocess the testing data
                test_data = preprocessdata(test_data, numeric_columns, cutMissingRows=options['cutMissingRows'], threshold=options['cut threshold'], inf=options['inf'], outliers=options['outliers'], N=options['outliers_N'])

                # Quantile Transformer
                if 'Quantile Transformer' in keys_list:
                    print("Quantile Transformer")
                    qt = outcome_dic['Quantile Transformer']
                    test_data[numeric_columns] = qt.transform(test_data[numeric_columns])
                
                    
                #Seperate the inputs and outputs for test data
                try:
                    X_test, y_test = split(test_data, input_columns, outcome_name)
                except:
                    st.error("Input Columns do not match with data set.")
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
                    
                # Imputing Data
                if 'Imputer' in keys_list:
                    print("Impute")
                    imputer = outcome_dic['Imputer']
                    X_test = pd.DataFrame(imputer.transform(X_test), columns = X_col)
                    y_test.reset_index(drop=True, inplace=True)
        
                # Scaling Data
                if 'Scaler' in keys_list:
                    print("Scaling")
                    scaler = outcome_dic['Scaler']
                    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
                    y_test.reset_index(drop=True, inplace=True)

                # Normalize the data
                if 'Normalizer' in keys_list:
                    print("Normalize")
                    normalizer = outcome_dic['Normalizer']
                    X_test[numeric_columns] = normalizer.transform(X_test[numeric_columns])
                    y_test.reset_index(drop=True, inplace=True)


            #print(X_test)
            #print(y_test)
            
            #st.write('Testing Started...')
            with st.spinner(f"Testing Model for {algo_name} on {o_name}..."):

                probas_test = model.predict_proba(X_test[features]) # get probablities with test set
                #model.predict(X_test[features]) # predict with test set
                
                # create a states able for metric on the test set
                res, res_array = full_roc_curve(y_test, probas_test[:, 1], index=cutoff_index)
                print("Results Array (Test Set): ", res)

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
                print("Test Accuracy:", (test_acc * 100))
                metric_dic['Accuracy'] = test_acc
                
                # training set results
                metric_dic['AUROC Score (Train)'] = df_res_train['auc']
                metric_dic['AUROC CI Low (Train)'] = df_res_train['auc_cilow']
                metric_dic['AUROC CI High (Train)'] = df_res_train['auc_cihigh']
                metric_dic['P (Train)'] = df_res_train['P'].astype(float)
                metric_dic['N (Train)'] = df_res_train['N'].astype(float)
                
                # Extract TP, FP, TN, FN
                metric_dic['TP'] = res['TP']
                metric_dic['FP'] = res['FP']
                metric_dic['TN'] = res['TN']
                metric_dic['FN'] = res['FN']
                
                metric_dic['P'] = res['P'].astype(float)
                metric_dic['N'] = res['N'].astype(float)
                
                metric_dic['precision'] = res['precision']
                metric_dic['recall'] = res['recall']
                
                metric_dic['Ground Truths'] = y_test.to_list()
                metric_dic['Predictions'] = predictions_test
                metric_dic['Probability Scores'] = probas_test.tolist()
                
                print(metric_dic)

            #st.write('Testing Complete!')
            with st.spinner(f"Testing on Data Done. Now saving ROC and SHAP charts for {algo_name} on {o_name}..."):
                #st.write('Results Saving...')
                # plot the ROC Curves
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
                plt.savefig(filename_roc,dpi=700)
                plt.show()  # Display the plot
                plt.close()
                
                # plot the SHAP Values
                plt.title(f'SHAP Values for {outcome_name} on {algo_name}')
                explainer = shap.Explainer(model.predict, X_test[features])
                #shap_values = explainer.shap_values(X_test[features])
                shap_values = explainer(X_test[features])
                
                try:
                    shap.summary_plot(shap_values, X_test[features], plot_type='dot', max_display = 10, show=False) 
                except:
                    shap.summary_plot(shap_values, X_test[features], plot_type='dot', show=False)
                
                fig = plt.gcf()  # Get current figure

                # Save the figure to a buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                image_data = buf.read()

                results_dictonary[algo_name][outcome_name] = {}

                results_dictonary[algo_name][outcome_name]['evaluation'] = metric_dic
                results_dictonary[algo_name][outcome_name]['shap values'] = image_data
                
                filename_shap = os.path.join(algorithm_folder, algo_name + "_" + sanitize_filename(outcome_name) + "_shap.png")
                plt.savefig(filename_shap,dpi=700)
                plt.show()  # Display the plot
                plt.close(fig)

                #st.write('Results Saved!')

            st.success(f"✅Testing Done for {algo_name} on {o_name}!")
            print("_________________________________________________________")

    st.success(f"✅Testing is Complete!")
    return results_dictonary  

def generate_results_table(results_dictonary):
    # Initialize an empty list to collect rows
    rows = []
    no_results = []
    
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
        
# --- Page Configuration ---
st.set_page_config(
    page_title="(Sklearn) Upload and Test ML Model",
    page_icon="🧪",
    layout="wide"
)
# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Testing_Models_Options.py")  # Redirect to the main back

# Title
st.title("🧪 Upload and Test ML Model (Sklearn)")

st.write("Test multiple models based on the Sklearn/Native framework using new/unseen data.")

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
outcomes = exp_dic['outcomes'] if exp_dic is not None else None
train_data_patth = exp_dic['train_data_path'] if exp_dic is not None else None

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
        st.write(f'**Train Data Path:** {train_data_patth}')
        st.write(f'**Time Created:** {time_created}')


# --- Step 2: Upload Data (Dynamic UI) ---
st.header("Step 2: Upload Data")

#user chooses whatever to upload the data or retrive a past data set from the database
data_options = st.radio("Choose an option:", ["Upload a testing set", "Retrive testing set from database"])

if data_options == "Upload a testing set":
    data_name_test = None
    # File uploader for the test
    uploaded_test_set = st.file_uploader("Upload a Testing Data Set")

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
            is_subset = all(x in test_cols for x in outcomes)
            if is_subset == False:
                st.error("Uploaded test set does not have the required output variables.")

        except Exception as e:
            st.error(f"Error loading file: {e}")
            is_subset = False
else:
    uploaded_test_set = None
    # Dropdown to select the testing dataset
    data_name_test = st.selectbox("Select a Testing Dataset from the database:", data_names_list_test, index=None, placeholder="Select One...")
    if data_name_test:
        # upload the testing set
        test_set = upload_data(os.path.join("Data Sets",data_name_test))
        # Replace inf and -inf with NaN
        test_set = test_set.replace([np.inf, -np.inf], np.nan)

        # check if upload test set has the require input and output variables
        test_cols = test_set.columns.to_list()
        # Check if outcomes is a subset of test_cols
        is_subset = all(x in test_cols for x in outcomes)
        if is_subset == False:
            st.error("Uploaded test set does not have the required output variables.")
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
#directory_name = st.text_input("Enter Directory name:", "results")
#os.makedirs(directory_name, exist_ok=True)

# Let user specify file name
#file_name = st.text_input("Enter File Name", "results")

if model_path is not None and (uploaded_test_set or data_name_test) is not None and len(selected_outcomes)!=0 and is_subset==True:
    # --- Step 4: Execute ---
    st.header("Step 4: Begin Testing")

    #st.write(db.results.find_one({"exp_name": exp_name, "test set": test_set_name}))

    # button to test the models
    if st.button('Test the models 🧪'):

        if db.results.find_one({"exp_name": exp_name, "test set": test_set_name}) == None:

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
                st.info(f"Testing Dataset of the same name is already in the database", icon="ℹ️")

                test_name = uploaded_test_set.name if data_name_test == None else data_name_test
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

            #try:
            # call the function to test models
            results_dictonary = test_models(models_dic, configuration, selected_algos, selected_outcomes, input_variables, test_set, exp_name, test_set_name, cutoff_index=threshold_type)

            if results_dictonary == "Error": # if test_models function returns an error
                st.error(f"❌ Testing '{exp_name}' Failed. Go back and check testing configuration.")
            else:
                with st.spinner("Saving all results..."): # if no error, then save the results
                    algorithm_folder = os.path.join("Results", exp_name, test_set_name)
                    os.makedirs(algorithm_folder, exist_ok=True)  # Create folder for algorithm results

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
                    results_dic
                    # get the current time
                    current_datetime = datetime.now()
                    current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                    # get test dataset name
                    test_name = uploaded_test_set.name if data_name_test == None else data_name_test
                    result = {
                        "exp_name": exp_name,
                        "type": model_type,
                        "test set": test_set_name,
                        "results_dic": results_dictonary,
                        "results_table": results_dic,
                        'dataset used': test_name,
                        "time_created": current_time
                    }

                    results.insert_one(result) # insert one dictonary
                    #except:
                        #st.write("Error has occured in testing. Check if feature set in data contains the required input variables from the model.")

                st.success(f"✅ Testing '{exp_name}' completed successfully!")
                st.subheader("Jump to Visualizing Results") # redirect to the testing section
                st.page_link("pages/Visualize_Options.py", label="Visualize Results", icon="📊")
        else:
            st.error("Test Result of the same name already exists. Please change Test Set name.")

#if os.path.isfile(f"{exp_name}_results.xlsx") or os.path.isfile(f"{exp_name}_results.joblib"):
    # swithces to Visualize_Results
    #if st.button('Visualize ML Results'):
        #st.switch_page("pages/Visualize_Options.py")  # Redirect to visualize_results.py
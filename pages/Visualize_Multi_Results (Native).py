import time
import io
import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
import plotly.express as px
#np.random.seed(1000)
rstate = 12
from PIL import Image
import os
import joblib
import datetime
import pprint
import pymongo
from pymongo import MongoClient

# import module
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

# --- Page Configuration ---
st.set_page_config(
    page_title="(Sklearn) Visualize ML Results",
    page_icon="📊",
    layout="wide"
)

# back button to return to main page
if st.button('Back'):
    st.session_state.df_total = pd.DataFrame()
    exp_names = None
    results_dict = None
    st.session_state.outcome_dic_total = {}
    list_of_outcomes = []
    outcome_options= []
    outcome_dic = None
    st.session_state.outcome_options = []
    st.switch_page("pages/Visualize_Options.py")  # Redirect to the main back

# Title
st.title("🔥 Visualize ML Results (Sklearn)")
st.write("This page helps you visualize the results of your ML model(s).")

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

# Initialize session state for df_total
if "outcome_dic_total" not in st.session_state:
    st.session_state.outcome_dic_total = {}

# Initialize the list of outcome options
if "outcome_options" not in st.session_state:
        st.session_state.outcome_options = []

def plot_roc(data, options):
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))

    # go throught each outcome to plot its ROC curve
    for option in options:
        rest, outcome = option.rsplit('-', 1)
        name, algo = rest.rsplit('-', 1)
        exp_name, test_set = name.rsplit('-', 1)
        #st.write(outcome)
        #st.write(algo)
        #st.write(test_set)
        #st.write(exp_name)
        if "evaluation" in list(data[exp_name][test_set][algo][outcome].keys()):
            values = data[exp_name][test_set][algo][outcome]["evaluation"] # get the ground truths and probs for the specified outcome
        else: 
            values = data[exp_name][test_set][algo][outcome]

        y_true = [int(x) for x in values["Ground Truths"]]
        y_prob = [x[1] for x in values["Probability Scores"]]

        # calcaute the AUROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        y_true_np = np.array(y_true)
        y_prob_np = np.array(y_prob)

        res, res_array = full_roc_curve(y_true_np, y_prob_np)

        # Extract Confidence Interval values
        try:
            auc_ci_low = values["AUROC CI Low"]
            auc_ci_high = values["AUROC CI High"]
        except: # calcuate them if not aviavble in data
            auc_ci_low = res['auc_cilow']
            auc_ci_high = res['auc_cihigh']

        # get other needed metrics
        specificity = res_array['tnr']
        sensitivity = res_array['tpr']

        # plot the ROC Curve
        ax.plot(fpr, tpr, label=f'{option} (AUC = {roc_auc:.4f} [{auc_ci_low:.4f}, {auc_ci_high:.4f}])', linewidth=2)

        # get the CI
        ax.fill_between(1-specificity, res_array['tpr_low'], res_array['tpr_high'], alpha=.2)


    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)

    st.pyplot(fig, use_container_width=False)  # Use Streamlit's function to display the plot


def plot_pr(data, options):
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))

    # go throught each outcome to plot its PR curve
    for option in options:
        rest, outcome = option.rsplit('-', 1)
        name, algo = rest.rsplit('-', 1)
        exp_name, test_set = name.rsplit('-', 1)

        if "evaluation" in list(data[exp_name][test_set][algo][outcome].keys()):
            values = data[exp_name][test_set][algo][outcome]["evaluation"] # get the ground truths and probs for the specified outcome
        else: 
            values = data[exp_name][test_set][algo][outcome]

        y_true = [int(x) for x in values["Ground Truths"]]
        y_prob = [x[1] for x in values["Probability Scores"]]

        # get the precision and recall values need to plot the curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc_score = auc(recall, precision)

        # plot the PR Curve
        ax.plot(recall, precision, label=f'{option} (AUC = {pr_auc_score:.4f})', linewidth=2)

    ax.plot([0, 1], [1, 0], 'k--')  # Diagonal line for reference
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curve", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)

    st.pyplot(fig, use_container_width=False)  # Use Streamlit's function to display the plot




def plot_shap(data, options):

    # Dropdown to select the exp to plot shap values with
    option = st.selectbox("Select an experiment", options)

    rest, outcome = option.rsplit('-', 1)
    name, algo = rest.rsplit('-', 1)
    exp_name, test_set = name.rsplit('-', 1)
    #st.write(outcome)
    #st.write(algo)
    #st.write(test_set)
    #st.write(exp_name)

    image_data = data[exp_name][test_set][algo][outcome]['shap values'] # get the shap image

    # Plot SHAP chart
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axis('off')  # Hides the axes

    # Load the image from binary data
    image = Image.open(io.BytesIO(image_data))
    plt.imshow(image)
    plt.show()
    st.pyplot(fig, use_container_width=False)
    plt.close()

def plot_confusion_matrix(data, option_cm):
    # Dropdown to select the exp to plot shap values with
    option_cm = st.selectbox("Select an experiment for CM", option_cm)
    rest, outcome = option_cm.rsplit('-', 1)
    name, algo = rest.rsplit('-', 1)
    exp_name, test_set = name.rsplit('-', 1)

    if "evaluation" in list(data[exp_name][test_set][algo][outcome].keys()):
        values = data[exp_name][test_set][algo][outcome]["evaluation"] # get the ground truths and probs for the specified outcome
    else: 
        values = data[exp_name][test_set][algo][outcome]

    # get ground truth and predictions
    y_true = [int(x) for x in values["Ground Truths"]]
    y_pred = [int(x) for x in values["Predictions"]]
    classes = sorted(set(y_true + y_pred))  # ensure all classes are included

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    cm = confusion_matrix(y_true_np, y_pred_np)
    #st.write(cm)

    # Plot Confusion matrix chart
    fig, ax = plt.subplots(figsize=(8, 6))

    # Show the matrix with color
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    # Set labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12)
    ax.set_ylabel('True Labels', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=12)

    # Set tick labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Rotate the tick labels and set alignment
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.show()
    st.pyplot(fig, use_container_width=False)
    plt.close()


# connect to database
client = MongoClient('10.14.1.12', 27017)

# create the database if it does not already exists
db = client.machine_learning_database

# create tables for models in the databse
models = db.models

# create the results collection if it does not already exists
results = db.results

# get all unique exp. names from results collection
exp_names = db.results.distinct("exp_name", {"type": "Native"})

st.write("")  # Add for more space
st.write("")

# Dropdown to select the experiment to display results from
exp_name = st.selectbox("Select a saved ML experiment", exp_names, help="Select a ML experiment from the database.")

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the Results Table</h2>", unsafe_allow_html=True)

# get the results_dicts
results_dicts = results.find({"exp_name": exp_name})

test_sets = [doc["test set"] for doc in results_dicts if "test set" in doc]
#st.write(test_sets)

# Dropdown to select the test set to display results from
test_set = st.selectbox("Select the test set used", test_sets, help="Select a specfic test result from the ML experiment.")

# get the final results_dict
results_dict = results.find_one({"exp_name": exp_name, "test set": test_set})

# get some info
model_type = results_dict['type'] if results_dict is not None else None
time_created = results_dict['time_created'] if results_dict is not None else None

# get the name of the data set used
try:
    test_set_name = results_dict['dataset used'] if results_dict is not None else None
except:
    test_set_name = 'N/A'

# give some information about the ML Test Result
with st.expander("▶️ ML Test Result Info"):
    # write all test content in expander
    st.markdown("##### **Name:**") 
    st.markdown(f'##### <u>{exp_name}</u>', unsafe_allow_html=True)
    st.markdown('##### **Test:**')
    st.markdown(f'##### <u>{test_set}</u>', unsafe_allow_html=True)
    st.write(f'**Model Type:** {model_type}')
    st.write(f'**Test Data Used:** {test_set_name}')
    st.write(f'**Time Created:** {time_created}')



# Initialize session state for df_total
if "df_total" not in st.session_state:
    st.session_state.df_total = pd.DataFrame()

if results_dict is not None and st.button('➕ Add Results', help="Add the result to the collective table."):
    # Get the outcome_dic for each outcome
    outcome_dic = results_dict['results_dic'] if results_dict is not None else None

    # check if results_dict is Native
    if outcome_dic is not None and results_dict['type'] != 'Native':
        st.write("Results is not Native.")
        results_dict = None
        outcome_dic = None

    try:
        # get the results table
        df = pd.DataFrame(results_dict['results_table'])

        # Combine the two columns into a single string
        df["Outcome_Algorithm"] = df["Outcome"] + " - " + df["Algorithm"]


        # add a column to dataframe with the name of the experiment
        df.insert(0, "Test Set", test_set)
        df.insert(0, "Exp_Name", exp_name)
        df["Exp_Name-Test Set"] = f'{exp_name}-{test_set}'

        # add error margins for AUROC Scores
        df["Upper_CI_Gap"] = df["AUROC CI Upper"] - df["AUROC Score"]
        df["Lower_CI_Gap"] = df["AUROC Score"] - df["AUROC CI Lower"]

        try:
            df["Upper_CI_Gap (Train)"] = df["AUROC CI Upper (Train)"] - df["AUROC Score (Train)"]
            df["Lower_CI_Gap (Train)"] = df["AUROC Score (Train)"] - df["AUROC CI Lower (Train)"]
        except:
            st.write("No Train Data results exists")

        df["Data Set used"] = test_set_name

        # Prevent adding the same file multiple times after rerun
        if st.session_state.df_total.empty or f'{exp_name}-{test_set}' not in st.session_state.df_total["Exp_Name-Test Set"].values:
             # add to dataframe
            st.session_state.df_total = pd.concat([st.session_state.df_total, df])

    except Exception as e:
        st.error(f"Error loading file: {e}")

else:
    outcome_dic = None

# button to remove a specific experiment from the overall table and outcome_dic_total
if st.button("➖ Remove Result", help="Remove the result to the collective table."):
    st.session_state.df_total = st.session_state.df_total[~(st.session_state.df_total["Exp_Name-Test Set"] == f'{exp_name}-{test_set}')] # remove from table
    
    if exp_name in list(st.session_state.outcome_dic_total.keys()) and test_set in list(st.session_state.outcome_dic_total[exp_name]):
        del st.session_state.outcome_dic_total[exp_name][test_set] # remove from outcome_dic_total
        if len(st.session_state.outcome_dic_total[exp_name]) == 0:
            del st.session_state.outcome_dic_total[exp_name]

    # edit the ROC plot to remove all plots that involved the removed result
    st.session_state.outcome_options = [item for item in st.session_state.outcome_options if f'{exp_name}-{test_set}' not in item]

# button to reset the table
if st.button('❌ Clear All Data', help="Clear the collective table."):
    st.session_state.df_total = pd.DataFrame()
    exp_names = None
    results_dict = None
    st.session_state.outcome_dic_total = {}
    list_of_outcomes = []
    outcome_options= []
    st.session_state.outcome_options = []
    outcome_dic = None

if len(st.session_state.df_total) != 0:
    # Display the DataFrame
    st.write("### ML Results Table:")
    st.dataframe(st.session_state.df_total)

      # List of all column names excluding the specified ones
    columns_to_exclude = ["Exp_Name-Test Set", 'Test Set', 'Algorithm', 'Exp_Name', 'Outcome', 'AUROC CI Lower', 'AUROC CI Upper', 'AUROC CI Lower (Train)', 'AUROC CI Upper (Train)', 
                          'Cutoff value', 'Best Model', 'TN', 'TP', 'FN', 'FP', 'P', 'N', 'P (Train)', 'N (Train)',
                          'Upper_CI_Gap', 'Lower_CI_Gap', "Upper_CI_Gap (Train)", "Lower_CI_Gap (Train)", "AUROC CI Upper (Train)", "AUROC CI Lower (Train)", "Outcome_Algorithm"]
    options = [col for col in st.session_state.df_total.columns if col not in columns_to_exclude]
    algorithms = ['All'] + st.session_state.df_total['Algorithm'].unique().tolist()
    outcomes = ['All'] + st.session_state.df_total['Outcome'].unique().tolist()
        
    # Dropdown to select the metric to disply in a barchart
    metric = st.selectbox("Select a Metric", options, help="Select a metric to view performace.")

    # choose which varaible to compare with
    variable = st.radio("Choose an option:", ["Algorithm", "Outcome"], help="Chose a specific algorithm or outcome to view its performace in all test results.")

    # error bars
    error_y = None # default
    error_y_minus = None

    if metric == "AUROC Score":
        error_y = "Upper_CI_Gap"
        error_y_minus = "Lower_CI_Gap"

    elif metric == "AUROC Score (Train)":
        error_y = "Upper_CI_Gap (Train)"
        error_y_minus = "Lower_CI_Gap (Train)"

    # Dropdown to select an algorithim
    choice = st.selectbox(f"Select a(n) {variable}", algorithms if variable == "Algorithm" else outcomes)

    if choice == 'All':
        metrics_data = st.session_state.df_total.copy()
    else:
        metrics_data = st.session_state.df_total[st.session_state.df_total[variable] == choice].copy()

    fig = px.bar(
        metrics_data,
        x="Outcome_Algorithm",
        y=metric,
        color="Exp_Name-Test Set",
        barmode="group",
        title=f"{metric} for Each Outcome",
        text_auto=True,  # Adds values on top of bars
        error_y=error_y,
        error_y_minus=error_y_minus
    )

    # Set y-axis range from 0 to 1
    fig.update_layout(yaxis=dict(range=[0, 1]))

    # Rotate text labels to be horizontal and limit decimals to 2 places
    fig.update_traces(
        textangle=0,  # Horizontal text
        texttemplate="%{y:.3f}",  # Format values to 2 decimal places
        textposition="inside"

    )


    # Add hover tooltip to show confidence intervals

    # Show the plot
    st.plotly_chart(fig)

st.write("")  # Add for more space
st.write("")

# Visuize and compare ROC and P-R Curves
st.markdown("<h2 style='text-align: center;'>Visualize the ROC and P-R Curve</h2>", unsafe_allow_html=True, help="Select a specific ML experiment, test set, algorithim, and outcome to plot its ROC and P-R curve.")

# add outcome_dic to st.session_state.outcome_dic_total
if (len(list(st.session_state.outcome_dic_total.keys())) == 0 or (exp_name not in list(st.session_state.outcome_dic_total.keys())) or test_set not in list(st.session_state.outcome_dic_total[exp_name].keys())) and outcome_dic is not None:
    
    if exp_name not in list(st.session_state.outcome_dic_total.keys()):
        st.session_state.outcome_dic_total[exp_name] = {}

    st.session_state.outcome_dic_total[exp_name][test_set] = outcome_dic

# Initialize session state variable
if "show_values_outcome_dic" not in st.session_state:
        st.session_state.show_values_outcome_dic = False

# Button to display values of outcome_dic
if st.button('Display the Values', help="Display a dictionary of all the results uploaded."):
    st.session_state.show_values_outcome_dic = True  # Set state to show values

# Button to hide values (appears only when values are shown)
if st.session_state.show_values_outcome_dic:
    st.write(st.session_state.outcome_dic_total)
    if st.button('Hide the Values'):
        st.session_state.show_values_outcome_dic = False  # Reset state to hide values
        st.rerun()  # Refresh the page to update UI


if len(list(st.session_state.outcome_dic_total.keys())) != 0:
    first_item = list(st.session_state.outcome_dic_total)[0]
    #st.write(first_item)

    exp = st.selectbox("Select the experiment", list(st.session_state.outcome_dic_total))
    
    test = st.selectbox("Select the test set", list(st.session_state.outcome_dic_total[exp].keys()))

    # select which ROC curve to plot
    algo = st.selectbox("Select the algorithim", list(st.session_state.outcome_dic_total[exp][test].keys()))

    outcome = st.selectbox("Select the outcome", st.session_state.outcome_dic_total[exp][test][algo].keys())

    if outcome is not None:

        if st.button('Plot'):
            if f'{exp}-{test}-{algo}-{outcome}' not in st.session_state.outcome_options:
                st.session_state.outcome_options.append(f'{exp}-{test}-{algo}-{outcome}')

    # button to reset the table
    if st.button('Clear All Plots', help="Clear the plot chart."):
        st.session_state.outcome_options = []

    st.title("ROC Curve Analysis", help="View the receiver operating characteristic curve for each selected outcome result.")
    if st.session_state.outcome_options:
        plot_roc(st.session_state.outcome_dic_total, st.session_state.outcome_options)

    st.title("P-R Curve Analysis", help="View the precision-recall curve for each selected outcome result.")
    if st.session_state.outcome_options:
        plot_pr(st.session_state.outcome_dic_total, st.session_state.outcome_options)

    st.title("SHAP Value Analysis", help="View the SHAP chart for each ROC curve.")
        
    if st.session_state.outcome_options:
        plot_shap(st.session_state.outcome_dic_total, st.session_state.outcome_options)


    st.title("Confusion Matrix Analysis", help="View the confusion matrix for each ROC/P-R curve")
        
    if st.session_state.outcome_options:
        plot_confusion_matrix(st.session_state.outcome_dic_total, st.session_state.outcome_options)


    
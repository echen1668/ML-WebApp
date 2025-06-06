import time

import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
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
import plotly.express as px
#np.random.seed(1000)
rstate = 12

import os
import joblib
import datetime
import pprint
import pymongo
from pymongo import MongoClient

# import module
import streamlit as st

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

# Initialize session state for df_total
if "outcome_dic_total" not in st.session_state:
    st.session_state.outcome_dic_total = {}

def plot_roc(data, options):
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(12, 8))

    # go throught each outcome to plot its ROC curve
    for option in options:
        #st.write(option)
        name, outcome = option.rsplit('-', 1)
        #st.write(name)
        if "evaluation" in list(data[name][outcome].keys()):
            values = data[name][outcome]["evaluation"] # get the ground truths and probs for the specified outcome
        else: 
            values = data[name][outcome]

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
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves", fontsize=16)
    ax.legend(loc="lower right", fontsize=11)

    st.pyplot(fig)  # Use Streamlit's function to display the plot

def plot_shap(data, options):

    # Dropdown to select the exp to plot shap values with
    option = st.selectbox("Select an experiment", options)

    #st.write(option)
    name, outcome = option.rsplit('-', 1)

    feature_importance = data[name][outcome]["feature importance"] # get the feature importance values
    df_feature_importance = pd.DataFrame(feature_importance)
    st.write(df_feature_importance)

    # the abs value of feature importance values
    df_feature_importance['importance (abs)'] = df_feature_importance['importance'].abs()

    # plot the 10 top most important features
    fig = plt.figure(figsize=(8, 6))
    ax = df_feature_importance['importance (abs)'][:10].plot(kind='bar')
    plt.title(f'Feature Importance for {outcome} on {name} (Top 10)')

    # Add text labels
    for i, v in enumerate(df_feature_importance['importance (abs)'][:10]):
        ax.text(i, v + 0.0005, f'{v:.3f}', ha='center')

    # Set y-axis range from 0 to 1
    ax.set_ylim(0, df_feature_importance['importance (abs)'][0] + 0.01)

    
        
    # Adjust x-axis labels for readability
    plt.xticks(rotation=45, ha='right')  # Rotate and align right
        
    st.pyplot(fig)  # Use Streamlit's function to display the plot

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
exp_names = db.results.distinct("exp_name", {"type": "AutoGluon"})

# back button to return to main page
if st.button('Back'):
    st.session_state.df_total = pd.DataFrame()
    exp_names = None
    results_dict = None
    st.session_state.outcome_dic_total = {}
    list_of_outcomes = []
    outcome_options= []
    outcome_dic = None
    st.switch_page("pages/Visualize_Options.py")  # Redirect to the main back

# Title
st.title("Visualize ML Results (AutoGluon)")

st.write("This page helps you visualize the results of your ML model(s).")
st.write("")  # Add for more space
st.write("")

# Dropdown to select the experiment to display results from
exp_name = st.selectbox("Select a saved ML experiment", exp_names)

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the Results Table</h2>", unsafe_allow_html=True)

# get the results_dicts
results_dicts = results.find({"exp_name": exp_name})

test_sets = [doc["test set"] for doc in results_dicts if "test set" in doc]
#st.write(test_sets)

# Dropdown to select the experiment to display results from
test_set = st.selectbox("Select the test set used", test_sets)

if st.button('Add Results'):
    # get the final results_dict
    results_dict = results.find_one({"exp_name": exp_name, "test set": test_set})

else:
    results_dict = None

# Get the outcome_dic for each outcome
outcome_dic = results_dict['results_dic'] if results_dict is not None else None

# check if results_dict is AutoGulon
if outcome_dic is not None and results_dict['type'] != 'AutoGluon':
    st.write("Results is not AutoGulon.")
    results_dict = None
    outcome_dic = None

# Initialize session state for df_total
if "df_total" not in st.session_state:
    st.session_state.df_total = pd.DataFrame()

if results_dict is not None:
    try:
        # # get the results table
        df = pd.DataFrame(results_dict['results_table'])

        # add a column to dataframe with the name of the experiment and test set name
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

        # Prevent adding the same file multiple times after rerun
        if st.session_state.df_total.empty or f'{exp_name}-{test_set}' not in st.session_state.df_total["Exp_Name-Test Set"].values:
             # add to dataframe
            st.session_state.df_total = pd.concat([st.session_state.df_total, df])

    except Exception as e:
        st.error(f"Error loading file: {e}")

# button to reset the table
if st.button('Clear All Data'):
    st.session_state.df_total = pd.DataFrame()
    exp_names = None
    results_dict = None
    st.session_state.outcome_dic_total = {}
    list_of_outcomes = []
    outcome_options= []
    outcome_dic = None

# Display the DataFrame
if len(st.session_state.df_total) != 0:
    st.write("### ML Results Table:")
    st.dataframe(st.session_state.df_total)

      # List of all column names excluding the specified ones
    columns_to_exclude = ["Exp_Name-Test Set", 'Test Set', 'Algorithm', 'Exp_Name', 'Outcome', 'AUROC CI Lower', 'AUROC CI Upper', 'AUROC CI Lower (Train)', 'AUROC CI Upper (Train)', 
                          'Cutoff value', 'Best Model', 'TN', 'TP', 'FN', 'FP', 'P', 'N', 'P (Train)', 'N (Train)',
                          'Upper_CI_Gap', 'Lower_CI_Gap', "Upper_CI_Gap (Train)", "Lower_CI_Gap (Train)", "AUROC CI Upper (Train)", "AUROC CI Lower (Train)"]
    options = [col for col in st.session_state.df_total.columns if col not in columns_to_exclude]
        
    # Dropdown to select the metric to disply in a barchart
    metric = st.selectbox("Select a Metric", options)

    # error bars
    error_y = None # default
    error_y_minus = None

    if metric == "AUROC Score":
        error_y = "Upper_CI_Gap"
        error_y_minus = "Lower_CI_Gap"

    elif metric == "AUROC Score (Train)":
        error_y = "Upper_CI_Gap (Train)"
        error_y_minus = "Lower_CI_Gap (Train)"

    fig = px.bar(
        st.session_state.df_total,
        x="Outcome",
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

# Visuize and compare ROC Curves
st.markdown("<h2 style='text-align: center;'>Visualize the ROC Curve</h2>", unsafe_allow_html=True)

# add outcome_dic to st.session_state.outcome_dic_total
if (len(list(st.session_state.outcome_dic_total.keys())) == 0 or f'{exp_name}-{test_set}' not in list(st.session_state.outcome_dic_total.keys())) and outcome_dic is not None:
    st.session_state.outcome_dic_total[f'{exp_name}-{test_set}'] = outcome_dic


if len(list(st.session_state.outcome_dic_total.keys())) != 0:
    first_item = list(st.session_state.outcome_dic_total)[0]
    #st.write(first_item)
            
    list_of_outcomes = [
                f"{exp}-{outcome}"
                for exp, outcomes in st.session_state.outcome_dic_total.items()
                for outcome in outcomes.keys()
    ]
else:
    list_of_outcomes = []
            
#st.write(list_of_outcomes)

# Initialize session state variable
if "show_values_outcome_dic" not in st.session_state:
        st.session_state.show_values_outcome_dic = False

# Button to display values of outcome_dic
if st.button('Display the Values'):
    st.session_state.show_values_outcome_dic = True  # Set state to show values

# Button to hide values (appears only when values are shown)
if st.session_state.show_values_outcome_dic:
    st.write(st.session_state.outcome_dic_total)
    if st.button('Hide the Values'):
        st.session_state.show_values_outcome_dic = False  # Reset state to hide values
        st.rerun()  # Refresh the page to update UI


st.title("ROC Curve Analysis")

# Select multiple outcomes for the ROC curve plot
outcome_options = st.multiselect("Select Outcomes to Plot", list_of_outcomes)

if outcome_options:
    plot_roc(st.session_state.outcome_dic_total, outcome_options)

st.title("Feature Importance Analysis")
        
if outcome_options:
    plot_shap(st.session_state.outcome_dic_total, outcome_options)
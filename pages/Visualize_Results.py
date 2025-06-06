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
#np.random.seed(1000)
rstate = 12

# import module
import streamlit as st

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

def plot_roc(data, outcomes):
    # Plot ROC curves
    fig, ax = plt.subplots(figsize=(12, 8))

    # go throught each outcome to plot its ROC curve
    for outcome in outcomes:
        if "evaluation" in list(data[outcome].keys()):
            values = data[outcome]["evaluation"] # get the ground truths and probs for the specified outcome
        else: 
            values = data[outcome]

        y_true = values["Ground Truths"]
        y_prob = values["Probability Scores"]

        # calcaute the AUROC
        fpr, tpr, _ = roc_curve(y_true, y_prob[1])
        roc_auc = auc(fpr, tpr)

        res, res_array = full_roc_curve(y_true.to_numpy(), y_prob[1].to_numpy())

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
        ax.plot(fpr, tpr, label=f'{outcome} (AUC = {roc_auc:.4f} [{auc_ci_low:.4f}, {auc_ci_high:.4f}])', linewidth=2)

        # get the CI
        ax.fill_between(1-specificity, res_array['tpr_low'], res_array['tpr_high'], alpha=.2)

    ax.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves", fontsize=16)
    ax.legend(loc="lower right", fontsize=14)

    st.pyplot(fig)  # Use Streamlit's function to display the plot

# back button to return to main page
if st.button('Back'):
    st.switch_page("pages/Visualize_Options.py")  # Redirect to the main back

# Title
st.title("Visualize ML Results")

st.write("This page helps you visualize the results of your ML model(s).")
st.write("")  # Add for more space
st.write("")

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the Results Table</h2>", unsafe_allow_html=True)

# File uploader for ML Results table
uploaded_file = st.file_uploader("Upload a results table. (Must be either an Excel or CSV file)")

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Display the DataFrame
        st.write("### ML Results Table:")
        st.dataframe(df)
        
        # List of all column names excluding the specified ones
        columns_to_exclude = ['Outcome', 'AUROC CI Lower', 'AUROC CI Upper', 'AUROC CI Lower (Train)', 'AUROC CI Upper (Train)', 'Cutoff value', 'Best Model', 'TN', 'TP', 'FN', 'FP', 'P', 'N', 'P (Train)', 'N (Train)']
        options = [col for col in df.columns if col not in columns_to_exclude]
        
        # Dropdown to select the metric to disply in a barchart
        metric = st.selectbox("Select a Metric", options)

        # Plotting the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        if metric == "AUROC Score":
            bars = ax.bar(df['Outcome'], df[metric], color='skyblue', yerr=[df["AUROC Score"]-df['AUROC CI Lower'], df['AUROC CI Upper']-df["AUROC Score"]], capsize=5)
        elif metric == "AUROC Score (Train)":
            bars = ax.bar(df['Outcome'], df[metric], color='skyblue', yerr=[df["AUROC Score (Train)"]-df['AUROC CI Lower (Train)'], df['AUROC CI Upper (Train)']-df["AUROC Score (Train)"]], capsize=5)
        else:
            bars = ax.bar(df['Outcome'], df[metric], color='skyblue')

        # Add values on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, 0.02, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=10)

        # Set y-axis range from 0 to 1
        ax.set_ylim(0, 1)

        ax.set_xlabel('Outcome')
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} for Each Outcome")
        ax.set_xticklabels(df['Outcome'], rotation=45, ha='right')

        # Show the plot
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading file: {e}")

st.write("")  # Add for more space
st.write("")

# Upload and look at a results table
st.markdown("<h2 style='text-align: center;'>Visualize the ROC Curve</h2>", unsafe_allow_html=True)

# File uploader for ML Results table
uploaded_file = st.file_uploader("Upload the Groud Truths and Probablites.")

if uploaded_file is not None:
    try:
        # Determine file type and read accordingly
        outcome_dic = joblib.load(uploaded_file)
        
        # Initialize session state variable
        if "show_values_outcome_dic" not in st.session_state:
            st.session_state.show_values_outcome_dic = False

        # Button to display values of outcome_dic
        if st.button('Display the Values'):
            st.session_state.show_values_outcome_dic = True  # Set state to show values

        # Button to hide values (appears only when values are shown)
        if st.session_state.show_values_outcome_dic:
            st.write(outcome_dic)
            if st.button('Hide the Values'):
                st.session_state.show_values_outcome_dic = False  # Reset state to hide values
                st.rerun()  # Refresh the page to update UI


        st.title("ROC Curve Analysis")

        # Select multiple outcomes for the ROC curve plot
        outcomes = st.multiselect("Select Outcomes to Plot", list(outcome_dic.keys()))

        if outcomes:
            plot_roc(outcome_dic, outcomes)
    except Exception as e:
        st.error(f"Error loading file: {e}")
import time
import io
import pandas as pd
import numpy as np
import sklearn as scikit_learn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
import shutil
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
import difflib
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

from Common_Tools import upload_data

# connect to database
client = MongoClient('10.14.1.12', 27017)
# create the database if it does not already exists
db = client.machine_learning_database
# create tables for models in the databse
models = db.models
# create the results collection if it does not already exists
results = db.results
# create the results if it does not already exists
datasets = db.datasets
# get all unique exp. names from results collection
exp_names_results = db.results.distinct("exp_name")
# get all unique exp. names from models collection
exp_names_models = db.models.distinct("exp_name")
# get all unique exp. name from both model and results collection
exp_names = list(set(exp_names_models + exp_names_results))
exp_names.sort()
# get all testing data names from database
data_names = db.datasets.distinct("data_name")

# --- Page Configuration ---
st.set_page_config(
    page_title="Database List",
    page_icon="🗒️",
    layout="wide"
)

# Custom CSS for background color
st.markdown("""
    <style>
    .custom-container {
        background-color: #ADD8E6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back


# --- Page Content ---

st.title("🗒️ Database List")
st.markdown("This page allows you to view a list of either all experiments done or datasets saved that are stored in the database.")

st.divider()

experiments, datasetlist = st.tabs(["🧪 Experiments", "📁 Datasets"])

##########################
# FOR EXPERIMENTS SECTION
##########################


# list all experiments
experiments.subheader("📋 All Experiments")
experiments.write("View all experiments stored in the database and view their infromation.")

# Dropdown to select the sorting method
#sort_method = experiments.selectbox("Sort By:", ['None', 'Alphabetically', 'Time'])

# Search bar to filter list by substring
search_substring = experiments.text_input("**Search**", help="Search for specific ML experiment by name.")
filtered_exp_names = [s for s in exp_names if search_substring in s]
#experiments.write(filtered_exp_names)

# Dropdown to select the type filter
filter_type = experiments.selectbox("**Select a type**", ['All', 'Native', 'Native-CV', 'AutoGulon'], help="Search a specific type of ML experiment.")

# list all ML Experiments
for exp_name in filtered_exp_names:

    # get model
    model = models.find_one({"exp_name": exp_name})

    # get the model type and number of algorithims
    if model==None: # for the CV types
        model = results.find_one({"exp_name": exp_name})
        model_type = model['type']
        num_algo = len(model['results_dic'])
    elif model['type'] =="Native":
        model_type = model['type']
        num_algo = len(model['algorithms'])
    else:
        model_type = model['type']
        num_algo = 'N/A'

    # get the time created
    try:
        time_created = model['time_created']
    except:
        time_created = 'N/A'

    # get all results for that ML experiment
    all_results = list(results.find({"exp_name": exp_name}))
    num_results = len(all_results)
    
    # if type is not part of filtered list, it is skipped
    if filter_type != 'All' and model_type != filter_type:
        continue

    # create a new container
    container = experiments.container(border=True)

    # write all model content in container
    container.markdown('<div class="custom-container">', unsafe_allow_html=True)
    container.write("")
    container.markdown(f'##### <u>{exp_name}</u>', unsafe_allow_html=True)
    container.write("")

    container.write(f'**Model Type:** {model_type}')
    container.write(f'**Time Created:** {time_created}')
    container.write(f'**Number of Algorithms:** {num_algo}')

    # list of algorthims
    if model_type == "Native":
        algo_list = list(model['algorithms'])
        container.write(f'**Algorithms:** {algo_list}')
    elif model_type == "Native-CV":
        # get the results table
        df = pd.DataFrame(model['results_table'])
        algo_list = list(df['Algorithm'].unique())
        container.write(f'**Algorithms:** {algo_list}')
    else:
        container.write(f'**Algorithms:** AutoGluon Stack Models')


    container.write(f'**Number of Test Results:** {num_results}')
    #container.write(f'**Time Created:** {time_created}')

    container.write("")

    #with container.expander("▶️ Full Information"):
        
    if model_type != "Native-CV":
        with container.expander("▶️ Configuration"):
            configuration = model['configuration']
            st.write(configuration)

    with container.expander("▶️ Test Results"):
        test_result_names = results.distinct("test set", {"exp_name": exp_name})
        # Dropdown to select the test result
        result_name = st.selectbox(f"**Select a test result for {exp_name}**", test_result_names, index=None, placeholder="Select One...")

        if result_name:
            results_table = results.find_one({"exp_name": exp_name, "test set": result_name})
            df_results_table = pd.DataFrame(results_table['results_table'])
            st.write(df_results_table)

    container.markdown('</div>', unsafe_allow_html=True)




#######################
# FOR DATABASE SECTION
#######################


# list all databases
datasetlist.subheader("📋 All Datasets")
datasetlist.write("View all datasets saved in the database and view their infromation.")

# Search bar to filter list by substring
search_substring = datasetlist.text_input("**Search**", help="Search for specific dataset by name.")
filtered_data_names = [s for s in data_names if search_substring in s]

# Dropdown to select the type filter
filter_type = datasetlist.selectbox("**Select a type**", ['All', 'Train', 'Test'], help="Search a specific type of saved dataset.")

#st.write(filtered_data_names)

# list all ML Experiments
for data_name in filtered_data_names:

    #st.write(data_name)
    # get dataset
    dataset = datasets.find_one({"data_name": data_name})

    # get the dataset info
    data_type = dataset['type']
    time_saved = dataset['time_saved']
    data_path = dataset['data_path']

    
    # if type is not part of filtered list, it is skipped
    if filter_type != 'All' and data_type != filter_type:
        continue

    # create a new container
    container = datasetlist.container(border=True)

    # write all model content in container
    container.markdown('<div class="custom-container">', unsafe_allow_html=True)
    container.write("")
    container.markdown(f'##### <u>{data_name}</u>', unsafe_allow_html=True)
    container.write("")

    container.write(f'**Data Type:** {data_type}')
    container.write(f'**Time Saved:** {time_saved}')

    with container.expander("▶️ Full Data"):
        df = upload_data(data_path)
        st.write(df)
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
import requests
import uuid
# import module
import streamlit as st
import os
import joblib
from datetime import datetime
import pprint
import pymongo
from pymongo import MongoClient
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
from multiprocessing import Process
from streamlit_cookies_manager import EncryptedCookieManager
from Common_Tools import cleanup_stale_jobs, wrap_text_excel, expand_cell_excel, grid_excel, upload_data, save_data, sanitize_filename
from roctools import full_roc_curve, plot_roc_curve
from AutoGulon_Test_Worker import auto_gulon_worker

def main():
    #selected_outcomes = None

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
    jobs = db.jobs
    db.jobs.create_index("expires_at", expireAfterSeconds=0)
    cleanup_stale_jobs(db) # clean up an 'zombie jobs'

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

    # get all current jobs running and check their status
    jobs_list = list(db.jobs.find({}, {"_id": 0}).sort("created_at", -1))
    with st.expander("💼 Show All Job Status"):
        for job in jobs_list:
            st.markdown(f"### {job['exp_name']}")
            for key, value in job.items():
                st.write(f"**{key}**:", value)
            st.divider()

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

    if input_variables is not None and model_path is not None and (uploaded_test_set or data_name_test) is not None and len(selected_outcomes)!=0 and is_subset==True:
        # --- Step 4: Execute ---
        st.header("Step 4: Begin Testing")

        # button to test the models
        if st.button('Test the models 🧪'):

            if db.results.find_one({"exp_name": exp_name, "test set": test_set_name}) == None:
                job_id = str(uuid.uuid4())

                p = Process(
                    target=auto_gulon_worker,
                    args=(
                        job_id,
                        client_name,
                        exp_name,
                        models_dic,
                        model_type,
                        test_set,
                        data_name_test,
                        uploaded_test_set,
                        test_set_name,
                        input_variables,
                        selected_outcomes,
                        train_set,
                        threshold_type
                    ),
                    daemon=False,   # ⚠️ REQUIRED
                )
                p.start()

                st.success(f"Testing started (job {job_id})")
                st.subheader("Jump to Visualizing Results") # redirect to the testing section
                st.page_link("pages/Visualize_Multi_Results (AutoGluon).py", label="Visualize Results", icon="📊")
            else:
                st.error("Test Result of the same name already exists. Please change Test Set name.")

if __name__ == "__main__":
    main()
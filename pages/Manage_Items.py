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

# --- Page Configuration ---
st.set_page_config(
    page_title="Manage Experiments/Results/Datasets",
    page_icon="🔧",
    layout="wide"
)

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back


# --- Page Content ---

st.title("🔧 Manage Experiments, Results, and Datasets")
st.markdown("This page allows you to delete or rename old models/experiments, old results, and saved datasets.")

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
# create the results collection if it does not already exists
results = db.results
# create the results if it does not already exists
datasets = db.datasets
# get all unique exp. names from results collection
exp_names_results = db.results.distinct("exp_name")
# get all unique exp. names from results collection
exp_names_models = db.models.distinct("exp_name")
# get all testing data names from database
data_names = db.datasets.distinct("data_name")

#st.divider()

delete, rename = st.tabs(["🗑️ Delete Something?", "✏️ Rename Something?"])


########################
# FOR DELETE SECTION
########################

# session states for disabling delete model button
if 'run_delete_button' in st.session_state and st.session_state.run_delete_button == True:
    st.session_state.running_delete = True
else:
    st.session_state.running_delete = False

# delete an experiment
delete.subheader("🗑️-🧪 Delete an Experiment?")
delete.write("Find an experiment from the database and remove it and all its associated results from both the database and file system.")
# Dropdown to select the experiment to display results from
exp_name_model = delete.selectbox("Select a saved ML experiment to delete", list(set(exp_names_models + exp_names_results)), index=None, placeholder="Select One...")

if len(list(set(exp_names_models + exp_names_results))) != 0 and exp_name_model != None and delete.button("Delete Experiment", disabled=st.session_state.running_delete, key='run_delete_button'):
    # remove all instances of the experiment from the database
    models.delete_many({"exp_name": exp_name_model})
    results.delete_many({"exp_name": exp_name_model})

    # remove all instances of the experiment from the file system
    model_path = os.path.join("Models", exp_name_model)
    delete.write(model_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        delete.success("Experiment is deleted successfully from Models folder.")
    else:
        delete.error("Experiment not found in Models folder")

    results_path = os.path.join("Results", exp_name_model)
    delete.write(results_path)
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
        delete.success("Experiment is deleted successfully from Results folder.")
    else:
        delete.error("Experiment not found in Results folder")


delete.write("")
delete.write("")

# session states for disabling delete results button
if 'run_results_button' in st.session_state and st.session_state.run_results_button == True:
    st.session_state.running_results = True
else:
    st.session_state.running_results = False

# delete a results
delete.subheader("🗑️-📊 Delete a Results?")
delete.write("Find a result from the database and remove it from both the database and file system.")
# Dropdown to select the experiment to display results from
exp_name_result = delete.selectbox("Select a saved ML Result", exp_names_results, index=None, placeholder="Select One...")

# get the results_dicts
results_dicts = results.find({"exp_name": exp_name_result})
test_sets = [doc["test set"] for doc in results_dicts if "test set" in doc]

# Dropdown to select the experiment to display results from
test_set = delete.selectbox("Select the test set used", test_sets)

if len(list(set(exp_names_models + exp_names_results))) != 0  and exp_name_result != None and delete.button("Delete Result", disabled=st.session_state.running_results, key='run_results_button'):
    # remove all instances of the experiment from the database
    results.delete_many({"exp_name": exp_name_result, "test set": test_set})

    # remove all instances of the experiment from the file system
    results_test_path = os.path.join("Results", exp_name_result, test_set)
    delete.write(results_test_path)
    if os.path.exists(results_test_path):
        shutil.rmtree(results_test_path)
        delete.success("Result is deleted successfully from Results folder.")
    else:
        delete.error("Result not found in Results folder")


# session states for disabling delete datasets button
if 'run_datasets_button' in st.session_state and st.session_state.run_datasets_button == True:
    st.session_state.running_datasets = True
else:
    st.session_state.running_datasets = False


# delete a dataset
delete.subheader("🗑️-📒 Delete a Dataset?")
delete.write("Find a dataset from the database and remove it from both the database and file system.")
# Dropdown to select the experiment to display dataset from
data_name = delete.selectbox("Select a saved Dataset", data_names, index=None, placeholder="Select One...")

if len(list(data_names)) != 0 and data_name != None and delete.button("Delete Dataset", disabled=st.session_state.running_datasets, key='run_datasets_button'):
    dataset_item = datasets.find_one({"data_name": data_name})

    # remove all instances of the dataset from the file system
    model_path = dataset_item['data_path']
    delete.write(model_path)
    if os.path.exists(model_path):
        os.remove(model_path)
        delete.success("Dataset is deleted successfully from Data Sets folder.")
    else:
        delete.error("Dataset not found in Data Sets folder")

    datasets.delete_many({"data_name": data_name}) # delete dataset item from database


########################
# FOR RENAME SECTION
########################

# session states for disabling rename button
if 'run_rename_button' in st.session_state and st.session_state.run_rename_button == True:
    st.session_state.running_rename = True
else:
    st.session_state.running_rename = False

# rename an experiment
rename.subheader("✏️-🧪 Rename an Experiment?")
rename.write("Find an experiment from the database and rename it.")
# Dropdown to select the experiment to display results from
exp_name_model = rename.selectbox("Select a saved ML experiment to rename", list(set(exp_names_models + exp_names_results)), index=None, placeholder="Select One...")
# Let user specify the new name
new_exp_name = rename.text_input("Enter a new exp. name", "")

if len(list(set(exp_names_models + exp_names_results))) != 0 and exp_name_model != None and new_exp_name != "" and new_exp_name not in list(set(exp_names_models + exp_names_results)) and rename.button("Rename Experiment", disabled=st.session_state.running_rename, key='run_rename_button'):
    if exp_name_model in exp_names_models:
        st.write("Model Found!")
        models.update_one(
            {"exp_name": exp_name_model}, # Filter condition
            {"$set": { "exp_name": new_exp_name,
            "model_path": f"Models\{new_exp_name}\{new_exp_name}_models.joblib" }}
        )
        # get the results_dict
        results_dict = models.find_one({"exp_name": new_exp_name})
    else:
        # get the results_dict
        results_dict = results.find_one({"exp_name": exp_name_model})
        if results_dict is not None:
            st.write("Cross Validation Found")

    results.update_many(
        {"exp_name": exp_name_model}, # Filter condition
        {"$set": { "exp_name": new_exp_name }} # Update operation
    )

    # get model type
    model_type = results_dict['type']
    st.write(f'Model Type is: {model_type}')

    # rename all instances of the experiment from the file system
    old_model_path = os.path.join("Models", exp_name_model)
    new_model_path = os.path.join("Models", new_exp_name)
    #rename.write(old_model_path)
    #rename.write(new_model_path)
    if os.path.exists(old_model_path):
        
        if model_type != 'Native-CV':
            # rename the joblib model file
            try:
                os.rename(os.path.join(old_model_path, f"{exp_name_model}_models.joblib"), os.path.join(old_model_path, f"{new_exp_name}_models.joblib"))
            except PermissionError:
                shutil.copytree(os.path.join(old_model_path, f"{exp_name_model}_models.joblib"), os.path.join(old_model_path, f"{new_exp_name}_models.joblib"))
                shutil.rmtree(os.path.join(old_model_path, f"{exp_name_model}_models.joblib"))
        #else:
            # if model joblib file is not found, then it is most likey a Cross Val. experiment
            #rename.info("Model not found in Models folder. If your experiment is a cross validation exp. then you wouldn't have a models file.")
        
        # rename the model folder
        try:
            os.rename(old_model_path, new_model_path)
        except PermissionError:
            shutil.copytree(old_model_path, new_model_path)
            shutil.rmtree(old_model_path)

        rename.success("Experiment is successfully renamed in Models folder.")
    else:
        rename.error("Experiment not found in Models folder")

    old_results_path = os.path.join("Results", exp_name_model)
    new_results_path = os.path.join("Results", new_exp_name)
    rename.write(old_results_path)
    rename.write(new_results_path)
    if os.path.exists(old_results_path) and model_type != 'Native-CV':
        
        # rename the results files
        directory_contents = os.listdir(old_results_path)
        for item_name in directory_contents:
            # You can also iterate through the subdirectories specifically
            st.write(item_name)
            item_path = os.path.join(old_results_path, item_name)
            os.rename(os.path.join(item_path, f"{exp_name_model}_results.joblib"), os.path.join(item_path, f"{new_exp_name}_results.joblib"))
            os.rename(os.path.join(item_path, f"{exp_name_model}_results.xlsx"), os.path.join(item_path, f"{new_exp_name}_results.xlsx"))
            
        # rename the model results folder
        try:
            os.rename(old_results_path, new_results_path)
        except PermissionError:
            shutil.copytree(old_results_path, new_results_path)
            shutil.rmtree(old_results_path)
        rename.success("Experiment is successfully renamed in Results folder.")

    elif os.path.exists(old_results_path) and model_type == 'Native-CV':
        # rename the results files
        os.rename(os.path.join(old_results_path, f"{exp_name_model}_results.joblib"), os.path.join(old_results_path, f"{new_exp_name}_results.joblib"))
        os.rename(os.path.join(old_results_path, f"{exp_name_model}_results.xlsx"), os.path.join(old_results_path, f"{new_exp_name}_results.xlsx"))

        # rename the model results folder
        try:
            os.rename(old_results_path, new_results_path)
        except PermissionError:
            shutil.copytree(old_results_path, new_results_path)
            shutil.rmtree(old_results_path)
        rename.success("Experiment is successfully renamed in Results folder.")

    else:
        rename.error("Experiment not found in Results folder")

    # Reset exp_name_model
elif exp_name_model != None and new_exp_name != "" and new_exp_name in list(set(exp_names_models + exp_names_results)):
    rename.error("Experiment with that name already exists")

# session states for disabling rename button for datasets
if 'run_dataset_button' in st.session_state and st.session_state.run_dataset_button == True:
    st.session_state.running_dataset = True
else:
    st.session_state.running_dataset = False

# rename an experiment
rename.subheader("✏️-📒 Rename Dataset?")
rename.write("Find an saved dataset from the database and rename it.")
# Dropdown to select the dataset to display from
data_name = rename.selectbox("Select a saved dataset to rename", data_names, index=None, placeholder="Select One...")
# Let user specify the new name
new_data_name = rename.text_input("Enter a new dataset name (No need to add file extension)", "")

# add the exenstion to new data name
if data_name != None and new_data_name != "":
    if data_name.endswith('.xlsx'):
        full_new_data_name = new_data_name + ".xlsx"
        alt_new_data_name = new_data_name + ".csv" # we have this variable to check for any dataset that have the same name but differnt file name
    elif data_name.endswith('.csv'):
        full_new_data_name = new_data_name + ".csv"
        alt_new_data_name = new_data_name + ".xlsx"

#st.write(full_new_data_name)
#st.write(alt_new_data_name)

if len(list(data_names)) != 0 and data_name != None and new_data_name != "" and (full_new_data_name not in list(data_names) and alt_new_data_name not in list(data_names))  and rename.button("Rename Dataset", disabled=st.session_state.running_dataset, key='run_dataset_button'):

    #if data_name.endswith('.xlsx'):
        #full_new_data_name = new_data_name + ".xlsx"
    #elif data_name.endswith('.csv'):
        #full_new_data_name = new_data_name + ".csv"

    # update dataset in databse
    datasets.update_many(
        {"data_name": data_name}, # Filter condition
        {"$set": { "data_name": full_new_data_name,
        "data_path": f"Data Sets\{full_new_data_name}" }}
    )

    # update dataset in folder system
    old_model_path = f"Data Sets\{data_name}"
    new_model_path = f"Data Sets\{full_new_data_name}"
    if os.path.exists(old_model_path):
        
        # rename the Data Sets folder
        try:
            os.rename(old_model_path, new_model_path)
        except PermissionError:
            shutil.copytree(old_model_path, new_model_path)
            shutil.rmtree(old_model_path)

        rename.success("Dataset is successfully renamed in Data Sets folder.")
    else:
        rename.error("Dataset not found in Data Sets folder")

elif data_name != None and new_data_name != "" and (full_new_data_name in list(data_names) or alt_new_data_name in list(data_names)):
    rename.error("Dataset with that name already exists")
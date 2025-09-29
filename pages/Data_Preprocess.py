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
from datetime import datetime
rstate = 12
from PIL import Image
import os
import joblib
import pprint
import pymongo
from pymongo import MongoClient
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from Common_Tools import generate_configuration_file, generate_configuration_template, generate_results_table, generate_congfig_file, get_avg_results_dic, wrap_text_excel, expand_cell_excel, grid_excel, generate_all_idx_files, upload_data, load_data, save_data, data_prep, data_prep_train_set, parse_exp_multi_outcomes, setup_multioutcome_binary, refine_binary_outcomes, generate_joblib_model, check_same_feature_set

def remove_highly_correlated_features(df, threshold):
    df_corr = df.corr().abs()
    columns = np.full((df_corr.shape[0],), True, dtype=bool)
    for i in range(df_corr.shape[0]):
        for j in range(i+1, df_corr.shape[0]):
            if df_corr.iloc[i,j] >= threshold:
                if columns[j]:
                    columns[j] = False
    selected_columns = df.columns[columns]
    removed_columns = list(df.columns[~columns])
    #print("Removed Features: ", removed_columns)
    return df[selected_columns], selected_columns, removed_columns


# --- Page Configuration ---
st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="📝",
    layout="wide"
)

# initalize the session state for the training and testing sets
if "df_train" not in st.session_state:
    st.session_state.df_train = pd.DataFrame()
if "df_test" not in st.session_state:
    st.session_state.df_test = pd.DataFrame()

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back


# --- Page Content ---

st.title("📝 Data Preprocessing and Engineering")
st.markdown("This page allows you to upload a dataset and do  preprocessing and engineering measures on it before saving it in the database.")

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

# upload the required dataset (training set)
uploaded_file_train = st.file_uploader("(Required) Upload a Training Dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file_train:
    train_set = uploaded_file_train.name
    #st.write(train_set)
    df_train = load_data(train_set, uploaded_file_train)
    #st.write(df_train)
    train_col = list(df_train.columns)
else:
    df_train = None
    st.session_state.show_dataset = False

# upload the optional dataset (test set)
uploaded_file_test = st.file_uploader("(Optional) Upload a Testing Dataset (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file_test:
    test_set = uploaded_file_test.name
    #st.write(train_set)
    df_test = load_data(test_set, uploaded_file_test)
    #st.write(df_test)
    test_col = list(df_test.columns)

    if df_train is not None and train_col != test_col:
        st.error("Testing Set does not have the same columns as the training set.")
        df_test = None
        st.session_state.show_dataset = False
else:
    df_test = None
    st.session_state.show_dataset = False


# Initialize session state variable
if "show_dataset" not in st.session_state:
        st.session_state.show_dataset = False

# Button to display values of outcome_dic
if df_train is not None and st.button('Display dataset'):
    st.session_state.show_dataset = True  # Set state to show dataset

# Button to hide datset (appears only when values are shown)
if st.session_state.show_dataset:
    st.header("Training Set")
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        st.write(df_train)
    else:
        st.write(st.session_state.df_train)

    if df_test is not None:
        st.header("Test Set")
        if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
            st.write(df_test)
        else:
            st.write(st.session_state.df_test)

    if st.button('Hide dataset'):
        st.session_state.show_dataset = False  # Reset state to hide dataset
        st.rerun()  # Refresh the page to update UI

# --- Remove specific features ---
st.header("Remove specific features")
if df_train is not None:
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        train_col = list(df_train.columns)
    else:
        train_col = list(st.session_state.df_train.columns)
    cols_to_remove = st.multiselect("Select a set of features to remove", train_col)

if df_train is not None and st.button('Remove Features'):
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        df_train = df_train.drop(cols_to_remove, axis=1)
        #st.write(df_train)
        st.session_state.df_train = df_train
    else:
        st.session_state.df_train = st.session_state.df_train.drop(cols_to_remove, axis=1)

    if df_test is not None:
        if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
            df_test = df_test.drop(cols_to_remove, axis=1)
            check_same_feature_set(df_train, df_test)
            st.session_state.df_test = df_test
        else:
            st.session_state.df_test = st.session_state.df_test.drop(cols_to_remove, axis=1)
            check_same_feature_set(st.session_state.df_train, st.session_state.df_test)

    st.success("Action Successful")



# --- Remove all features that have too many missing values ---
st.header("Remove all features that have too many missing values")
# any columns that have less than threshold% of rows with non-null values will be removed
min_threshold = st.number_input("Enter the threshold of mininum rows with non-null values required for a feature to remain (Ex: 0.80 mean all columns with below 80 percent of non-missing values will be removed):", min_value=0.0, max_value=1.0, value=0.8)

if df_train is not None and st.button('Remove Missing Values'):
    #try:
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        df_train_old = df_train.copy()
        df_train = df_train.dropna(axis=1, thresh=min_threshold)
        new_cols = list(df_train.columns)
        st.session_state.df_train = df_train

        # get all rows that are removed
        removed_cols = list(set(df_train_old.columns) - set(df_train.columns))
    else:
        df_train_old = st.session_state.df_train.copy()
        st.session_state.df_train = st.session_state.df_train.dropna(axis=1, thresh=min_threshold)
        new_cols = list(st.session_state.df_train.columns)

         # get all rows that are removed
        removed_cols = list(set(df_train_old.columns) - set(st.session_state.df_train.columns))


    if df_test is not None:
        if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
            df_test = df_test[new_cols]
            check_same_feature_set(df_train, df_test)
            st.session_state.df_test = df_test
        else:
            st.session_state.df_test = st.session_state.df_test[new_cols]
            check_same_feature_set(st.session_state.df_train, st.session_state.df_test)

    # print all rows that are removed
    st.write("Removed columns:", removed_cols)
    st.success("Action Successful")
    #except:
        #st.error("Action Failed to Complete")


# --- Remove highly correlated features ---
st.header("Remove highly correlated features")
unique_value_threshold = st.number_input("Enter the minimum unique value threshold for a input variable to be consired catagorical:", min_value=1, max_value=100, value=10)
corr_threshold = st.number_input("Enter the threshold for the required correlation:", min_value=0.0, max_value=1.0, value=0.9)

if df_train is not None and st.button('Remove HC features'):
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        df_train_old = df_train.copy()
    else:
        df_train_old = st.session_state.df_train.copy()

    # Identify discrete numeric columns as categorical if they have fewer unique values than the threshold
    discrete_numeric_cols = [col for col in df_train_old.select_dtypes(include=np.number).columns if len(df_train_old[col].unique()) < unique_value_threshold]
    # Exclude the identified discrete numeric columns from numeric_cols
    numeric_cols = [col for col in df_train_old.select_dtypes(include=np.number).columns if col not in discrete_numeric_cols]
    numeric_df_train = df_train_old[numeric_cols]

    with st.spinner("Removing Highly Correlated Features..."):
        # Pass DataFrame and Threshold value 
        _, selected_columns, removed_columns = remove_highly_correlated_features(numeric_df_train,corr_threshold)
        st.write("Removed Features: ", list(removed_columns))
        if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
            df_train.drop(columns=removed_columns, inplace=True)
            st.session_state.df_train = df_train
            new_cols = list(df_train.columns)
        else:
            st.session_state.df_train.drop(columns=removed_columns, inplace=True)
            new_cols = list(st.session_state.df_train.columns)

        if df_test is not None:
            if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
                df_test = df_test[new_cols]
                check_same_feature_set(df_train, df_test)
                st.session_state.df_test = df_test
            else:
                st.session_state.df_test = st.session_state.df_test[new_cols]
                check_same_feature_set(st.session_state.df_train, st.session_state.df_test)

    st.success("Action Successful")


    # --- Remove low variance features ---
st.header("Remove low variance features")

var_threshold = st.number_input("Enter the variance threshold:", min_value=0.0, max_value=1.0, value=0.2)

if df_train is not None and st.button('Remove LV features'):
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        df_train_old = df_train.copy()
    else:
        df_train_old = st.session_state.df_train.copy()

    # Identify discrete numeric columns as categorical if they have fewer unique values than the threshold
    discrete_numeric_cols = [col for col in df_train_old.select_dtypes(include=np.number).columns if len(df_train_old[col].unique()) < unique_value_threshold]
    # Exclude the identified discrete numeric columns from numeric_cols
    numeric_cols = [col for col in df_train_old.select_dtypes(include=np.number).columns if col not in discrete_numeric_cols]
    numeric_df_train = df_train_old[numeric_cols]

    # remove the columns
    columns_to_remove = list(numeric_df_train.columns[(np.nanstd(numeric_df_train, axis=0) < var_threshold)])
    st.write("Removed Features: ", columns_to_remove)
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        df_train.drop(columns=columns_to_remove, inplace=True)
        st.session_state.df_train = df_train
        new_cols = list(df_train.columns)
    else:
        st.session_state.df_train.drop(columns=columns_to_remove, inplace=True)
        new_cols = list(st.session_state.df_train.columns)

    if df_test is not None:
        if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
            df_test = df_test[new_cols]
            check_same_feature_set(df_train, df_test)
            st.session_state.df_test = df_test
        else:
            st.session_state.df_test = st.session_state.df_test[new_cols]
            check_same_feature_set(st.session_state.df_train, st.session_state.df_test)
    
    st.success("Action Successful")


st.write("")
st.write("")
st.write("")
st.write("")

# --- Save Dataset ---
st.header("Save the Data")
if df_train is not None:
    train_set_name = st.text_input("Enter Name of the Train Set (No need to add file extension)", train_set.removesuffix(".xlsx") if train_set.endswith('.xlsx') else train_set.removesuffix(".csv"))
else:
    train_set_name = "train_set.csv"

if df_test is not None:    
    test_set_name = st.text_input("Enter Name of the Test Set (No need to add file extension)", test_set.removesuffix(".xlsx") if test_set.endswith('.xlsx') else test_set.removesuffix(".csv"))
else:
    test_set_name = "test_set.csv"

if df_train is not None and st.button("Save dataset(s)"):

    if train_set.endswith('.xlsx'):
        train_set_name = train_set_name + ".xlsx"
    elif train_set.endswith('.csv'):
        train_set_name = train_set_name + ".csv"

    # save the training set
    # get the current time
    current_datetime = datetime.now()
    current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # get the most updated data set
    if st.session_state.df_train.empty == True or (not(set(st.session_state.df_train.columns).issubset(df_train.columns)) and (len(st.session_state.df_train) != len(df_train))):
        st.write("upload orginal")
        #st.write(st.session_state.df_train.empty)
        #st.write()
        saved_train_set = df_train.copy()
    else:
        st.write("upload new")
        saved_train_set = st.session_state.df_train.copy()

    # save train set into data folder and database
    os.makedirs("Data Sets", exist_ok=True)
    save_data(train_set_name, saved_train_set, os.path.join("Data Sets", train_set_name))
    dataset_train = {
            "data_name": train_set_name,
            "type": "Train",
            "time_saved": current_time,
            "data_path": os.path.join("Data Sets", train_set_name),
            "exps used": []
    }
    if train_set_name not in data_names:
        datasets.insert_one(dataset_train)
        st.success(f"Training Dataset {train_set_name} is saved in the database", icon="ℹ️")
    else:
        datasets.replace_one({"data_name": train_set_name}, dataset_train)
        st.info(f"Training Dataset {train_set_name} of the same name is overwritten in the database", icon="ℹ️")

    # save the testing set
    if df_test is not None:
    
        # get the current time
        current_datetime = datetime.now()
        current_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        if test_set.endswith('.xlsx'):
            test_set_name = train_set_name + ".xlsx"
        elif test_set.endswith('.csv'):
            test_set_name = train_set_name + ".csv"

        # get the most updated data set
        if st.session_state.df_test.empty == True or (not(set(st.session_state.df_test.columns).issubset(df_test.columns)) and (len(st.session_state.df_test) != len(df_test))):
            saved_test_set = df_test.copy()
        else:
            saved_test_set = st.session_state.df_test.copy()

        # save train set into data folder and database
        os.makedirs("Data Sets", exist_ok=True)
        save_data(test_set_name, saved_test_set, os.path.join("Data Sets", test_set_name))
        dataset_test = {
                "data_name": test_set_name,
                "type": "Test",
                "time_saved": current_time,
                "data_path": os.path.join("Data Sets", test_set_name),
                "exps used": []
        }

        if test_set_name not in data_names:
            datasets.insert_one(dataset_test)
            st.success(f"Testing Dataset {test_set_name} is saved in the database", icon="ℹ️")
        else:
            datasets.replace_one({"data_name": test_set_name}, dataset_test)
            st.info(f"Testing Dataset {test_set_name} of the same name is overwritten in the database", icon="ℹ️")

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

from pymongo import MongoClient
import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
from Common_Tools import cleanup_stale_jobs, wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve


# --- Page Configuration ---
st.set_page_config(
    page_title="Visualize ML Results",
    page_icon="📊",
    layout="wide"
)

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
client = MongoClient(client_name, 27017)

# create the database if it does not already exists
db = client.machine_learning_database
# create the results if it does not already exists
jobs = db.jobs
db.jobs.create_index("expires_at", expireAfterSeconds=0)
cleanup_stale_jobs(db) # clean up an 'zombie jobs'

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back

# get all current jobs running and check their status
jobs_list = list(db.jobs.find({}, {"_id": 0}).sort("created_at", -1))
with st.expander("💼 Show All Job Status"):
    for job in jobs_list:
        st.markdown(f"### {job['exp_name']}")
        for key, value in job.items():
            st.write(f"**{key}**:", value)
        st.divider()

# Title
st.title("📊 Visualize ML Results")

st.write("This page helps you visualize the results of your ML model(s).")

# two options to decide what to do
#left_column, middle_column, right_column = st.columns(3)

# Visualize ML Results button on AutoGluon models
st.subheader("🤖 Visualize the results of ML experiments done with AutoGluon")
st.page_link("pages/Visualize_Multi_Results (AutoGluon).py", label=":blue[Visualize AutoGulon Models]", icon="📊") # Redirect to Visualize_Multi_Results (AutoGluon).py
st.write("")

# Visualize ML Results button on Native/Sklearn models
st.subheader("🔥 Visualize the results of ML experiments done with Native (sklearn) models")
st.page_link("pages/Visualize_Multi_Results (Native).py", label=":blue[Visualize Sklearn Models]", icon="📊") # Redirect to Visualize_Multi_Results (Native).py
st.write("")

# Visualize ML Results button for CV 
st.subheader("❌ Visualize the results of ML experiments done with Cross Validation using Native (sklearn) models")
st.page_link("pages/Visualize_Multi_Results (CV).py", label=":blue[Visualize CV Sklearn Results]", icon="📊") # direct to Visualize_Multi_Results.py 
st.write("")
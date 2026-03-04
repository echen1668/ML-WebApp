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
    page_title="Train and Develop ML Model",
    page_icon="🚀",
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
st.title("Train and Develop ML Model")
st.write("This page allows you to train and develop your machine learning model.")

# AutoGulon Models
st.subheader("🤖 Train and Develop AutoGulon ML Models")
st.write("Use the AutoGulon framework to develop your experiment.")
st.page_link("pages/Training_AutoGulon_Models.py", label="Use AutoGulon", icon="🚀")


# Sklearn/Native Models
st.subheader("🔥 Train and Develop Sklearn ML Models")
st.write("Use the traditional/native Sklearn framework to develop your experiment.")
st.page_link("pages/Training_Models_Native.py", label="Use Sklearn", icon="🚀")
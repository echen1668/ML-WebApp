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
    page_title="Instructions Page",
    page_icon="📖",
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

st.title("📖 Instructions Page")
st.markdown("This page give you steps on how to use the ML Experimentation Hub.")

st.divider()


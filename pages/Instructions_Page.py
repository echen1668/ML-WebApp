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

st.subheader("🚀 Machine Learning Training")
st.write("This app allows you a generate a ML model or a set of ML Models by creating an ML experiment.")
st.write("This ML experiment allows the user to set up a custom configurtion on what datasets to use, what type of training to use, what variables to use, what algorithims, and/or what hyperparamter setup to use.")
st.write("Once custom configurtion is done, user submits it in order to do data preprocssing and ML training. All datasets, data preprocessing steps, and ML models created are saved.")
st.divider()

st.subheader("🧪 Machine Learning Testing")
st.write("One a model is generated, the user then can move to the testing section to test the ML model.")
st.write("User can upload a new dataset that has the required variables and can choose which outcomes and which classification threshold to test the model with.")
st.write("User must add a unique name to the test for it to be saved in the databse.")
st.divider()

st.subheader("🤖 Visualize Machine Learning Results")
st.write("After ML testing is done, the user can move to the visualiztion section to get a clear view of the test results.")
st.write("The user must select a test result from a selected ML experiment. This results in its results table being uploaded to a collective table in which the user can upload as many tables from as many test results as possible.")
st.write("User can view a group bar chart that shows the scores of a selected metric from each outcome, each algorithim, and each ML experiment. It can compare performaces from each experiment.")
st.write("User can filter by outcome or algorithim for the Sklearn version.")
st.write("The user can also plot mutiple ROC and P-R curves from any outcome from any uploaded test results on a chart and compare them.")
st.write("For each ROC/P-R curve, the user can also view its SHAP/feature importance chart and/or its confusion matrix chart.")
st.divider()

st.subheader("🗒️ Database List")
st.write("User can go to this section to view general infromation on each completed ML experiment and saved database, including what time they were saved.")
st.divider()

st.subheader("🔧 Manage Experiments, Results, and Datasets")
st.write("User can go to this section to delete or rename old models/experiments, old results, and saved datasets in which they have to select it.")
st.divider()


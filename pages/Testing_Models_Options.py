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

# import module
import streamlit as st

from Common_Tools import wrap_text_excel, expand_cell_excel, grid_excel
from roctools import full_roc_curve, plot_roc_curve

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back

# Title
st.title("🧪 Upload and Test ML Model")

st.write("This page allows you to upload and test your machine learning model.")
st.write("Load a previously trained model from database and evaluate its performance against a new, unseen dataset.")

# two options to decide what to do
left_column, right_column = st.columns(2)

with left_column:
    # AutoGulon Models
    st.subheader("🤖 Test AutoGulon ML Models")
    st.write("Upload and Test AutoGulon ML Models.")
    st.page_link("pages/Testing_AutoGulon_Models.py", label="Test AutoGulon Models", icon="🧪")


with right_column:
    # Sklearn/Native Models
    st.subheader("🔥 Test Sklearn ML Models")
    st.write("Upload and Test Sklearn (native) ML Models")
    st.page_link("pages/Testing_Native_Models.py", label="Test Sklearn Models", icon="🧪")

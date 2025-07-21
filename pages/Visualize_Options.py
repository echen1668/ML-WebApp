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

# --- Page Configuration ---
st.set_page_config(
    page_title="Visualize ML Results",
    page_icon="📊",
    layout="wide"
)

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back

# Title
st.title("📊 Visualize ML Results")

st.write("This page helps you visualize the results of your ML model(s).")

# two options to decide what to do
left_column, middle_column, right_column = st.columns(3)

# Visualize ML Results button on AutoGluon models
if left_column.button('🤖 Visualize the results of ML experiments done with AutoGluon'):
    st.switch_page("pages/Visualize_Multi_Results (AutoGluon).py")  # Redirect to Visualize_Multi_Results (AutoGluon).py

# Visualize ML Results button on Native/Sklearn models
if middle_column.button('🔥 Visualize the results of ML experiments done with Native (sklearn) models'):
    st.switch_page("pages/Visualize_Multi_Results (Native).py")  # Redirect to Visualize_Multi_Results (Native).py

# Visualize ML Results button for CV
if right_column.button('❌ Visualize the results of ML experiments done with Cross Validation using Native (sklearn) models'):
    st.switch_page("pages/Visualize_Multi_Results (CV).py")  # Redirect to Visualize_Multi_Results.py    
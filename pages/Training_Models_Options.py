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
    page_title="Train and Develop ML Model",
    page_icon="🚀",
    layout="wide"
)

# back button to return to main page
if st.button('Back'):
    st.switch_page("ML_Interface.py")  # Redirect to the main back

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
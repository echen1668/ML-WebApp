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
    st.switch_page("pages/Testing_Models_Options.py")  # Redirect to the main back

# Title
st.title("Sklearn/Native Models")

st.write("Choose which training mode you want to do.")

# two options to decide what to do
left_column, middle_column, right_column = st.columns(3)

if left_column.button('Model Generation (with dedicated test set(s))'):
    st.switch_page("pages/Training_Native_Models_W_TS.py")  # Redirect to Testing_Single_Models.py

if middle_column.button('Model Generation (only one data set used)'):
    st.switch_page("pages/Training_Native_Models_WO_TS.py")  # Redirect to Testing_Native_Models.py
    
if right_column.button('Cross Validation'):
    st.switch_page("pages/Training_Native_Models_CV.py")  # Redirect to Testing_Native_Models.py
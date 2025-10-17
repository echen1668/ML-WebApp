import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

from utils import data_utils, training_utils, helpers
from utils import excel_utils
from utils.training_utils import ALGO_MAP

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Create New Experiment")

def create_column_summary(df): # get a summry of the dataframe
    summary_list = []
    for col in df.columns:
        col_type = df[col].dtype
        missing_count = df[col].isnull().sum()
        missing_percent = f"{missing_count / len(df) * 100:.2f}%"
        unique_count = df[col].nunique()
        
        col_summary = {
            "Column": col, "Data Type": str(col_type), "Missing Values": missing_count,
            "Missing (%)": missing_percent, "Unique Values": unique_count
        }
        
        if np.issubdtype(col_type, np.number):
            stats = df[col].describe()
            col_summary.update({
                "Mean": f"{stats.get('mean', 0):.2f}", "Std Dev": f"{stats.get('std', 0):.2f}",
                "Min": stats.get('min', 0), "Max": stats.get('max', 0)
            })
        summary_list.append(col_summary)
    return pd.DataFrame(summary_list)

def create_experiment_summary(data_df, index_df): #create a summery of the experiment
    summary_list = []
    for exp_name, row in index_df.iterrows():
        inputs = [col for col, val in row.items() if val == 1]
        outcomes = [col for col, val in row.items() if val == 2]
        
        for outcome in outcomes:
            prevalence = "N/A"
            if outcome in data_df.columns and set(data_df[outcome].dropna().unique()).issubset({0, 1}):
                prev_rate = data_df[outcome].value_counts(normalize=True).get(1, 0)
                prevalence = f"{prev_rate * 100:.2f}%"
            summary_list.append({
                "Experiment Plan": exp_name, "Input Count": len(inputs),
                "Outcome": outcome, "Prevalence": prevalence
            })
    return pd.DataFrame(summary_list)

# --- Main Orchestration Function ---
def run_training(options):
    """
    The entire training workflow based on options.
    """
    st.write("---")
    st.subheader("🚀 Starting Experiment...")
    
    exp_name = options['exp_name']
    training_method = options['training_method']
    
    # --- 1. Setup Folders ---
    with st.spinner("Setting up project structure..."):
        project_folder = os.path.join("Models", exp_name)
        results_folder = os.path.join("Results", exp_name)
        os.makedirs(project_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)
        st.success(f"Created directories at `{project_folder}` and `{results_folder}`.")
        time.sleep(1)

    # --- 2. Load Data and Parse Schema ---
    with st.spinner("Loading data and parsing index file..."):
        try:
            index_df = data_utils.load_data(options['index_file'])
            
            # Use the correct key to get the uploaded file object
            main_uploaded_file = options.get('main_dataset_uploader')

            if not main_uploaded_file:
                 st.error("Main dataset not found in session state. Please re-upload.")
                 st.stop()

            if training_method == "Dedicated Test Set":
                # The uploader with the key 'main_dataset_uploader' holds the training set in this case
                train_df = data_utils.load_data(main_uploaded_file)
                test_df = data_utils.load_data(options['test_dataset'][0])
            else: # Train/Test Split or Cross-Validation
                train_df = data_utils.load_data(main_uploaded_file)
                test_df = None

            input_cols, output_cols = data_utils.parse_inputs_outputs(train_df, index_df)
            st.success("Data loaded and schema parsed successfully.")
            # ...
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
            
    # --- 3. Loop Through Algorithms and Execute Workflows ---
    all_experiment_results = {}
    for algo_full_name in options['algorithms']:
        algo_short_name = ALGO_MAP[algo_full_name]
        st.write("---")
        st.info(f"Running workflow for algorithm: **{algo_full_name}**")
        
        # This is the dispatcher logic that calls the correct backend function
        try:
            with st.spinner(f"Training {algo_full_name} with method '{training_method}'... This may take a while."):
                
                # We call the placeholder utils functions here.
                # In a real scenario, these would perform the heavy lifting.
                if training_method == "Dedicated Test Set":
                    algo_results = training_utils.run_training_workflow_dedicated_test(
                        df_train=train_df, df_test=test_df, input_cols=input_cols,
                        label_cols=output_cols, options=options, algorithm=algo_short_name,
                        project_folder=project_folder, project_name=exp_name
                    )
                elif training_method == "Train/Test Split":
                    algo_results = training_utils.run_training_workflow_traintest_split(
                        df=train_df, input_cols=input_cols, label_cols=output_cols,
                        options=options, algorithm=algo_short_name, project_folder=project_folder,
                        project_name=exp_name
                    )
                elif training_method == "Cross-Validation":
                    algo_results = training_utils.run_training_workflow_cv(
                        df=train_df, input_cols=input_cols, label_cols=output_cols,
                        options=options, algorithm=algo_short_name, project_folder=project_folder,
                        project_name=exp_name
                    )
            all_experiment_results[algo_short_name] = algo_results
            st.success(f"Successfully completed workflow for **{algo_full_name}**.")
            
        except Exception as e:
            st.error(f"An error occurred while training {algo_full_name}: {e}")
            continue # Move to the next algorithm

    # --- 4. Finalize Experiment ---
    st.write("---")
    with st.spinner("Finalizing experiment: saving metadata and final reports..."):
        # Save the final configuration to a database or file
        training_utils.archive_experiment(
        exp_name=exp_name,
        options=options,
        all_results=all_experiment_results,
        results_folder=results_folder
    )
        st.success(f"✅ Experiment '{exp_name}' completed successfully!")


# --- UI Initialization ---
st.title("🚀 Create a New Experiment")
st.markdown("Configure and launch a new model training workflow.")

# (The rest of the UI code is the same as before)
# ...
if 'training_method' not in st.session_state:
    st.session_state.training_method = "Train/Test Split"
    st.session_state.exp_name = ""
    st.session_state.algorithms = []

# --- Step 1: Experiment Setup ---
st.header("Step 1: Define Experiment and Data Strategy")
st.text_input("Experiment Name", key="exp_name", help="Enter a unique name for this experiment.")
st.radio(
    "Select Training Method",
    ["Train/Test Split", "Dedicated Test Set", "Cross-Validation"],
    key="training_method",
    horizontal=True,
    help="Choose how to evaluate model."
)

# --- Step 2: Upload Data (Dynamic UI) ---
st.header("Step 2: Upload Data")
col1, col2 = st.columns(2)
with col1:
    # Use a consistent key for the main dataset uploader
    dataset_uploader_key = "main_dataset_uploader"
    
    if st.session_state.training_method in ["Train/Test Split", "Cross-Validation"]:
        st.file_uploader("Upload Dataset (CSV or Excel)", type=['csv', 'xlsx'], key=dataset_uploader_key)
    else: # Dedicated Test Set
        st.file_uploader("Upload Training Dataset", type=['csv', 'xlsx'], key=dataset_uploader_key)
        st.file_uploader("Upload Testing Dataset", type=['csv', 'xlsx'], key="test_dataset", accept_multiple_files=True)

with col2:
    st.file_uploader("Upload **Completed** Index File", type=['xlsx'], key="index_file")


st.write("")

# Check if a main dataset has been uploaded
main_uploaded_file = st.session_state.get(dataset_uploader_key)
completed_index_file = st.session_state.get("index_file")

if main_uploaded_file and not completed_index_file:
    st.info("A dataset has been uploaded. Now generate a matching index file to edit.")
    
    index_file_bytes = excel_utils.generate_index_from_upload(main_uploaded_file)
    
    if index_file_bytes:
        original_filename = Path(main_uploaded_file.name).stem
        download_filename = f"{original_filename}_index_template.xlsx"
        
        st.download_button(
            label="⬇️ Download Generated Index File Template",
            data=index_file_bytes,
            file_name=download_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
# Add feedback to the user once they've uploaded their completed file.
elif main_uploaded_file and completed_index_file:
    #st.success("✅ Completed index file has been uploaded.")
    if st.button("📊 View Data Summary and Experiment Plan", use_container_width=True):
        # When button is clicked, open a dialog
        st.dialog("Data Summary")
        st.header("Experiment Plan Summary")
        st.markdown("Table: summarizes the inputs and outcomes.")
            
        # Load data and generate summaries INSIDE the dialog
        data_df = data_utils.load_data(main_uploaded_file)
        index_df = data_utils.load_data(completed_index_file)
            
        exp_summary_df = create_experiment_summary(data_df, index_df)
        st.dataframe(exp_summary_df, use_container_width=True)

        st.header("Dataset Column Analysis")
        st.markdown("Table: Statistics for each column in the dataset.")
        col_summary_df = create_column_summary(data_df)
        st.dataframe(col_summary_df, use_container_width=True)

        if st.button("Close"):
            st.rerun() # Closes the dialog

# --- Step 3: Configure Training Pipeline ---
st.header("Step 3: Configure Training Pipeline")
if st.session_state.get("main_dataset_uploader"): #st.session_state.get("main_dataset") or st.session_state.get("train_dataset"):
    
    st.multiselect(
        "Select ML Algorithms to Train",
        ['Random Forest', 'XGBoost', 'Cat Boost', "Logistic Regression L2", "SVM", "KNearest Neighbors"],
        key="algorithms"
    )
    with st.expander("▶️ Data Preprocessing Options"):
        st.radio("One-Hot Encode Categorical Features?", ("True", "False"), key="oneHotEncode", horizontal=True)
        st.radio("Impute Missing Values?", ("True", "False"), key="impute", horizontal=True)
        st.radio("Handle Infinite Values By:", ('replace with null', 'replace with zero'), key="inf_handling", horizontal=True)
        
        st.subheader("Handle Outliers")
        st.radio("Outlier Strategy:", ('None', 'remove rows', 'log'), key="outliers")
        if st.session_state.outliers != 'None':
            st.number_input("Outlier Threshold", value=50000, key="outliers_N")

        st.subheader("Handle Missing Rows")
        st.radio("Cut Rows with Missing Values?", ("True", "False"), key="cutMissingRows", horizontal=True)
        if st.session_state.cutMissingRows == 'True':
            st.slider("Minimum % of Non-Null Values Required", 0.0, 1.0, 0.6, key="cut_threshold")
            
    with st.expander("▶️ Feature Scaling & Transformation"):
        st.radio("Scale Numerical Features?", ("True", "False"), index=1, key="scaling", horizontal=True)
        if st.session_state.scaling == 'True':
            st.selectbox("Scaling Method", ['MinMaxScaler', 'RobustScaler', 'MaxAbsScaler', 'StandardScaler'], key="scalingMethod")
        
        st.radio("Apply Quantile Transformation?", ("True", "False"), index=1, key="quantileTransformer", horizontal=True)
        st.radio("Apply Normalization?", ("True", "False"), index=1, key="normalize", horizontal=True)

    with st.expander("▶️ Rebalancing & Feature Selection"):
        st.radio("Rebalance the Training Data?", ("True", "False"), index=1, key="rebalance", horizontal=True)
        if st.session_state.rebalance == 'True':
            st.selectbox("Rebalancing Method", ['RandomUnderSampler', 'RandomOverSampler', 'SMOTE', 'ADASYN'], key="rebalance_type")
        
        st.radio("Perform Feature Selection?", ("True", "False"), index=1, key="featureSelection", horizontal=True)
        if st.session_state.featureSelection == 'True':
            st.selectbox("Feature Selection Method", ['MRMR', 'SelectKBest-f_classif', 'SelectKBest-chi2'], key="featureSelection_method")
            st.number_input("Number of Features to Select", min_value=1, value=20, key="N_features")
    
    with st.expander("▶️ Training & Evaluation Options"):
        st.selectbox("Hyperparameter Search Strategy", ['random', 'bayesian', 'grid'], key="strategy")
        st.number_input("Number of Search Iterations", min_value=1, value=50, key="num_itr")
        st.number_input("Number of Cross-Validation Folds", min_value=2, value=5, key="k_fold")
        st.number_input("Number of CV Repeats", min_value=1, value=1, key="n_repeats")
        st.number_input("Minimum Positive Cases for an Outcome", min_value=1, value=10, key="min_positives")
        if st.session_state.training_method != "Cross-Validation":
            st.selectbox("Optimal Cutoff Threshold Method", ['youden', 'mcc', 'ji', 'f1'], key="threshold_type")


# --- Step 4: Execute ---
st.header("Step 4: Run Experiment")

if st.button("Start Training", type="primary", use_container_width=True):
    # Validation
    if not st.session_state.exp_name:
        st.error("Please enter an experiment name.")
    elif not st.session_state.algorithms:
        st.error("Please select at least one algorithm.")
    elif not st.session_state.index_file:
        st.error("Please upload an index file.")
    else:
        # Assemble all options from session_state into a single dictionary
        options = {key: val for key, val in st.session_state.items()}
        
        # Call the main orchestration function
        run_training(options)

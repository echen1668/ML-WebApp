import streamlit as st
import pymongo
from pymongo import MongoClient

# --- Page Configuration ---
st.set_page_config(
    page_title="ML Experimentation Hub",
    page_icon="🔬",
    layout="wide"
)
from streamlit_cookies_manager import EncryptedCookieManager
cookies = EncryptedCookieManager(prefix="mlhub_", password="some_secret_key")
if not cookies.ready():
    st.stop()

col1, col2, col3 = st.columns((3, 2, 0.5))

# --- Page Content ---
col1.title("🔬 ML Experimentation Hub")
col1.markdown("Training, testing, and visualizing machine learning models.")

if col3.button("Instructions Page", type="primary", icon='📖'):
    st.switch_page("pages/Instructions_Page.py") # switch to instructions page


st.divider()

# Read cookie
if "client_name" in cookies:
    st.session_state["client_name"] = cookies["client_name"]

# enter name of the IP address of a database. Ex: '10.14.1.12'
client_name = st.text_input("Enter Name of the MongoDB Database. (Ex: 10.14.1.12)", value=cookies.get("client_name", ""))
#client_name = st.text_input("Enter Name of the MongoDB Database. (Ex: 10.14.1.12)")
if client_name:
    st.session_state["client_name"] = client_name
    cookies["client_name"] = client_name
    cookies.save()

    #st.write(st.query_params["client_name"])
    # connect to database
    client = MongoClient(client_name, 27017, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command('ping')
        connected = True
    except Exception as e:
        st.error("Connection to database failed.")
        connected = False
else:
    connected = False

if connected == True:
    st.divider()
    st.header("Choose Your Workflow")

    # 1. Link to the Training Page
    st.subheader("🚀 Create a New Experiment")
    st.write("Configure and launch a new training job. Choose from multiple training strategies like Train/Test Split or Cross-Validation, select algorithms, and define preprocessing pipeline.")
    st.page_link("pages/Training_Models_Options.py", label="Go to Training", icon="🚀")

    st.write("")

    # 2. Link to the Testing Page
    st.subheader("🧪 Test an Existing Model")
    st.write("Load a previously trained model from database and evaluate its performance against a new, unseen dataset.")
    st.page_link("pages/Testing_Models_Options.py", label="Go to Testing", icon="🧪")

    st.write("")

    # 3. Link to the Visualization Page
    st.subheader("📊 Visualize & Compare Results")
    st.write("Load one or more experiments from the database to compare performance metrics, or upload local result files for manual visualization.")
    st.page_link("pages/Visualize_Options.py", label="Go to Visualization", icon="📊")

    st.write("")
    st.write("")
    st.write("")
    st.write("___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________")
    st.write("")
    st.write("")
    st.write("")

    # 4. Link to the Deletion Page
    st.subheader("🗒️ Database List")
    st.write("Look at a list of all the ML experiments and datasets saved and access of information about them.")
    st.page_link("pages/Experiment_Info.py", label="Go to List", icon="🗒️") 

    # 5. Link to the Deletion Page
    st.subheader("🔧 Manage Experiments, Results, and Datasets")
    st.write("Search for one or more experiments/results/dataset from the database and either remove it from both the database and file system or rename it.")
    st.page_link("pages/Manage_Items.py", label="Go to Management", icon="🔧")
else:
    st.info("Please connect to a database.")
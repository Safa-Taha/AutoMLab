import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import (
    setup as classification_setup,
    compare_models as classification_compare_models,
    pull as classification_pull,
    save_model as classification_save_model,
    load_model as classification_load_model,
    predict_model as classification_predict_model,
    tune_model as classification_tune_model,
)
from pycaret.regression import (
    setup as regression_setup,
    compare_models as regression_compare_models,
    pull as regression_pull,
    save_model as regression_save_model,
    load_model as regression_load_model,
    predict_model as regression_predict_model,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os

# Initialize Streamlit session state for analysis type and uploaded file
if "analysis_type" not in st.session_state:
    st.session_state.analysis_type = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Sidebar for navigation and information
with st.sidebar:
    st.image(
        "https://images.idgesg.net/images/article/2018/01/emerging-tech_ai_machine-learning-100748222-large.jpg"
    )
    st.title("AutoMLab")
    st.info("AutoMLab is an application allows you to build an automated machine learning pipeline using Streamlit and PyCaret !")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Data Processing", "ML", "Download", "Inference"])

# Function to validate the target variable
def validate_target_variable(df, target):
    return target in df.columns

# Function to validate features list
def validate_features(df, features):
    return [feature for feature in features if feature in df.columns]

# Function to check if a column needs conversion
def needs_conversion(column):
    if column.dtype == 'object':
        try:
            pd.to_numeric(column, errors='coerce')
            return True
        except:
            return False
    else:
        return False

# Upload section
if choice == "Upload":
    st.title("Upload Your Data for Modeling")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_file = uploaded_file.name
        df.to_csv("sourcedata.csv", index=False)
        
        # Convert problematic columns to float
        columns_to_convert = [column for column in df.columns if needs_conversion(df[column])]
        for column in columns_to_convert:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # Convert the DataFrame to an Arrow table
        table = pa.Table.from_pandas(df)
        
        # Write the table to a Parquet filepq.write_table(table, 'data.parquet')
        
        st.success("Data uploaded successfully!")

# Profiling section
elif choice == "Profiling":
    st.title("Automated Exploratory Data Analysis")
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv")
        df = df.sample(frac=1)
        profile_report = ProfileReport(df, title="Data Profiling Report")
        st_profile_report(profile_report)
    else:
        st.warning("No data available for profiling. Please upload a file.")

# Data Processing section
elif choice == "Data Processing":
    st.title("Data Processing")
    if os.path.exists("sourcedata.csv"):
        df = pd.read_csv("sourcedata.csv")
        
        # Drop columns if required
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if columns_to_drop:
            df = df.drop(columns_to_drop, axis=1)
        
        # Handle missing values in numerical columns
        missing_value_option = st.selectbox("Choose how to handle missing values", ["Mean", "Median", "Mode", "Drop rows"])
        if missing_value_option in ["Mean", "Median", "Mode"]:
            numeric_cols = df.select_dtypes(include='number').columns
            for col in numeric_cols:
                if missing_value_option == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif missing_value_option == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif missing_value_option == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_value_option == "Drop rows":
            df.dropna(inplace=True)
        
        # Encode categorical data
        encode_option = st.selectbox("Choose how to encode categorical data", ["Label Encoding", "One-Hot Encoding"])
        if encode_option == "Label Encoding":
            le = LabelEncoder()
            for col in df.select_dtypes(include='object').columns:
                df[col] = le.fit_transform(df[col])
        elif encode_option == "One-Hot Encoding":
            df = pd.get_dummies(df)
        
        # Save processed data
        df.to_csv("processed_data.csv", index=False)
        st.success("Data processed and saved successfully!")

# ML section
elif choice == "ML":
    st.title("Machine Learning")
    if os.path.exists("processed_data.csv"):
        df = pd.read_csv("processed_data.csv")
        
        # Print DataFrame columns for checking available columns
        st.write("DataFrame Columns:", df.columns)
        
        # Select target variable
        target_options = df.columns
        target = st.selectbox("Select Target Variable", target_options)
        
        # Validate selected target variable
        if not validate_target_variable(df, target):
            st.error(f"Error: The selected target variable '{target}' is not in the DataFrame columns.")
        else:
            # Choose features for modeling
            features = st.multiselect("Select Features", [col for col in df.columns if col != target])
            
            # Validate and filter the ignore_features list
            ignore_features = [col for col in df.columns if col not in features]
            ignore_features = [feature for feature in ignore_features if feature in df.columns]
            
            # Print ignore_features list to confirm the selected features
            st.write("Ignore Features:", ignore_features)
            
            # Proceed if features are selected
            if features:
                if st.button("Run Model"):
                    # Determine analysis type (classification or regression)
                    if df[target].dtype == "object" or len(df[target].unique()) < 10:st.session_state.analysis_type = "Classification"
                    else:
                        st.session_state.analysis_type = "Regression"
                    
                    # Model training and evaluation
                    if st.session_state.analysis_type == "Regression":
                        # Validate ignore_features to ensure it does not include the target variable
                        if target in ignore_features:
                            ignore_features.remove(target)
                        regression_setup(df, target=target, ignore_features=ignore_features)
                        setup_df = regression_pull()
                        st.info("ML Experiment Settings")
                        st.dataframe(setup_df)
                        best_model = regression_compare_models()
                        compare_df = regression_pull()
                        st.info("Model Comparison")
                        st.dataframe(compare_df)
                        regression_save_model(best_model, "best_model.pkl")
                        st.success("Model trained and saved successfully!")
                    
                    elif st.session_state.analysis_type == "Classification":
                        # Validate ignore_features to ensure it does not include the target variable
                        if target in ignore_features:
                            ignore_features.remove(target)
                        classification_setup(df, target=target, ignore_features=ignore_features)
                        setup_df = classification_pull()
                        st.info("ML Experiment Settings")
                        st.dataframe(setup_df)
                        best_model = classification_compare_models()
                        compare_df = classification_pull()
                        st.info("Model Comparison")
                        st.dataframe(compare_df)
                        tuned_model = classification_tune_model(best_model)
                        classification_save_model(tuned_model, "best_model.pkl")
                        st.success("Model trained and saved successfully!")
            else:
                st.warning("Please select features for modeling.")

# Download section
elif choice == "Download":
    st.title("Download the Best Model")
    model_file = "best_model.pkl"
    if os.path.exists(model_file):
        st.download_button("Download the Model", open(model_file, "rb"), file_name="best_model.pkl")
        st.success("Model downloaded successfully!")
    else:
        st.warning("No model available to download.")

# Inference section
elif choice == "Inference":
    st.title("Upload Data for Predictions")
    inference_file = st.file_uploader("Upload a CSV file for predictions", type=["csv"])
    if inference_file and st.session_state.analysis_type:
        df_inference = pd.read_csv(inference_file)
        
        # Load the best model
        model_file = "best_model.pkl"
        if st.session_state.analysis_type == "Regression":
            regression_model = regression_load_model(model_file)
            predictions = regression_predict_model(regression_model, data=df_inference)
        
        elif st.session_state.analysis_type == "Classification":
            classification_model = classification_load_model(model_file)
            predictions = classification_predict_model(classification_model, data=df_inference)
        
        # Display predictions and allow download
        st.subheader("Predictions:")
        st.write(predictions)
        predictions.to_csv("predictions.csv", index=False)
        st.download_button("Download Predictions", open("predictions.csv", "rb"), file_name="predictions.csv")
        st.success("Predictions saved and ready for download!")
    else:
        st.warning("Please upload a CSV file and ensure a model type is specified for predictions.")
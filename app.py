import pandas as pd
import numpy as np
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils import check_consistent_length

# Define the MODELS dictionary
MODELS = {
    'LinearRegression': LinearRegression,
    'Ridge': Ridge,
    'SVR': SVR,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'KNeighborsRegressor': KNeighborsRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'LogisticRegression': LogisticRegression,
    'SVC': SVC,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'KNeighborsClassifier': KNeighborsClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
}

# Define the REGRESSION_MODELS set
REGRESSION_MODELS = {
    'LinearRegression',
    'Ridge',
    'SVR',
    'DecisionTreeRegressor',
    'RandomForestRegressor',
    'KNeighborsRegressor',
    'GradientBoostingRegressor',
}

# Define the CLASSIFICATION_MODELS set
CLASSIFICATION_MODELS = {
    'LogisticRegression',
    'SVC',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'KNeighborsClassifier',
    'GradientBoostingClassifier',
}

# Define a function to validate the target variable
def validate_target_variable(df, target):
    """To validate that the target variable exists in the DataFrame."""
    return target in df.columns

# Define a function to display charts and metrics based on the selected method
def display_chart(df, target, method):
    # Split the data into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_target = train[target]
    test_target = test[target]
    train = train.drop(target, axis=1)
    test = test.drop(target, axis=1)

    # Get feature names from the train DataFrame
    feature_names = train.columns

    # Instantiate the selected model
    model = MODELS[method]()

    model.fit(train, train_target)

    # Predict on the test data
    preds = model.predict(test)

    # Verify data dimensions for plotting
    try:
        check_consistent_length(test_target, preds)
    except ValueError as e:
        st.markdown(f"**Warning:** {str(e)}")
        return

    # Display metrics based on the selected method
    if method in REGRESSION_MODELS:
        # Calculate metrics
        mse = mean_squared_error(test_target, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_target, preds)

        # Display metrics
        st.write("Metrics:")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R-Squared: {r2}")

        # Provide plot options for the user to choose
        plot_options = ['Actual vs Predicted Scatter Plot', 'Residual Plot']
        if method in ['LinearRegression', 'Ridge']:
            plot_options.append('Coefficient Plot')
        if method == 'Ridge':
            plot_options.append('Alpha vs Coefficient Plot')

        selected_plot = st.selectbox('Choose a Plot to Display:', plot_options)

        # Display the selected plot
        if selected_plot == 'Actual vs Predicted Scatter Plot':
            fig, ax = plt.subplots()
            ax.scatter(test_target, preds, color='blue')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Scatter Plot: {method}')
            st.pyplot(fig)

        elif selected_plot == 'Residual Plot':
            residuals = test_target - preds
            fig, ax = plt.subplots()
            ax.scatter(preds, residuals, color='blue')
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residual Plot: {method}')
            st.pyplot(fig)

        elif selected_plot == 'Coefficient Plot' and method in ['LinearRegression', 'Ridge']:
            try:
                coefficients = model.coef_
                
                # Handle potential scalar values (e.g., numpy.float64)
                if isinstance(coefficients, np.float64):
                    st.markdown("**Warning:** Coefficients not available for the selected model.")
                    return
                
                # Ensure coefficients is an array and lengths of feature_names and coefficients match
                if len(coefficients[0]) != len(feature_names):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                # Plot the coefficients
                fig, ax = plt.subplots()
                ax.bar(feature_names, coefficients[0], color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.set_title('Coefficient Plot')
                plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
                st.pyplot(fig)
            except (ValueError, AttributeError, TypeError) as e:
                st.markdown(f"**Warning:** {str(e)}")

        elif selected_plot == 'Alpha vs Coefficient Plot' and method == 'Ridge':
            # Handle alphas and coefficients arrays
            alphas = np.logspace(-3, 3, 7)
            coefs = []
            try:
                for alpha in alphas:
                    ridge_model = Ridge(alpha=alpha)
                    ridge_model.fit(train, train_target)
                    coefs.append(ridge_model.coef_)

                if len(coefs) > 0 and len(coefs[0]) != len(feature_names):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                # Plot alpha vs coefficient plot
                fig, ax = plt.subplots()
                for coef, alpha in zip(coefs, alphas):
                    ax.plot(feature_names, coef[0], label=f'Alpha: {alpha}')
                ax.set_title('Alpha vs Coefficient Plot')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.legend()
                plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** {str(e)}")

    elif method in CLASSIFICATION_MODELS:
        # Calculate metrics
        accuracy = accuracy_score(test_target, preds)
        precision = precision_score(test_target, preds, average='macro')
        recall = recall_score(test_target, preds, average='macro')
        f1 = f1_score(test_target, preds, average='macro')

        # Display metrics
        st.write("Metrics:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-Score: {f1}")

        # Confusion matrix flag to avoid duplicate plotting
        confusion_matrix_plotted = False
        
# Provide plot options for the user to choose
        plot_options = ['ROC Curve', 'Confusion Matrix']
        if method in ['LogisticRegression', 'SVC']:
            plot_options.append('Coefficient Plot')
        if method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            plot_options.append('Feature Importance Plot')
            plot_options.append('Partial Dependence Plot')
            plot_options.append('Learning Curve Plot')

        selected_plot = st.selectbox('Choose a Plot to Display:', plot_options)

        # Display the selected plot
        if selected_plot == 'ROC Curve':
            # Check if the classification problem is binary
            if len(np.unique(test_target)) != 2:
                st.markdown("**Warning:** ROC Curve can only be plotted for binary classification problems.")
                return
            try:
                # Plot ROC curve for logistic regression
                if method == 'LogisticRegression':
                    pred_probs = model.predict_proba(test)[:, 1]
                elif method == 'SVC':
                    decision_function = model.decision_function(test)
                    pred_probs = decision_function
                
                # Calculate and plot ROC curve
                fpr, tpr, _ = roc_curve(test_target, pred_probs)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label='ROC curve', color='blue')
                ax.plot([0, 1], [0, 1], linestyle='--', color='red')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve: {method}')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** Error plotting ROC Curve: {str(e)}")

        # Display the selected plot
        if selected_plot == 'ROC Curve':
            try:
                if method == 'LogisticRegression':
                    # Predict probabilities
                    if hasattr(model, 'predict_proba'):
                        pred_prob = model.predict_proba(test)[:, 1]
                    else:
                        pred_prob = model.decision_function(test)
                        
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(test_target, pred_prob)
                    
                    # Plot ROC curve
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label='ROC curve', color='blue')
                    ax.plot([0, 1], [0, 1], linestyle='--', color='red')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve: {method}')
                    ax.legend()
                    st.pyplot(fig)
                    
                elif method == 'SVC':
                    # Calculate decision function
                    decision_function = model.decision_function(test)
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(test_target, decision_function)
                    
                    # Plot ROC curve
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label='ROC curve', color='blue')
                    ax.plot([0, 1], [0, 1], linestyle='--', color='red')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve: {method}')
                    ax.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** ROC Curve unable to plot. {str(e)}")

        elif selected_plot == 'Confusion Matrix' and not confusion_matrix_plotted:
            # Plot confusion matrix only once
            cm = confusion_matrix(test_target, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
            ax.set_title(f'Confusion Matrix: {method}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)
            confusion_matrix_plotted = True

        elif selected_plot == 'Coefficient Plot' and method in ['LogisticRegression', 'SVC']:
            try:
                coefficients = model.coef_
                
                # Handle potential scalar values
                if isinstance(coefficients, np.float64):
                    st.markdown("**Warning:** Coefficients not available for the selected model.")
                    return
                
                # Check if lengths of feature_names and coefficients match
                if len(coefficients[0]) != len(feature_names):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                # Plot coefficients
                fig, ax = plt.subplots()
                ax.bar(feature_names, coefficients[0], color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.set_title('Coefficient Plot')
                plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
                st.pyplot(fig)
            except (ValueError, AttributeError, TypeError) as e:
                st.markdown(f"**Warning:** {str(e)}")

        elif selected_plot == 'Feature Importance Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            # Get feature importances
            try:
                feature_importances = model.feature_importances_
                
                # Plot feature importance plot
                fig, ax = plt.subplots()
                ax.bar(feature_names, feature_importances, color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance Plot')
                plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** Error in Feature Importance Plot: {str(e)}")

        elif selected_plot == 'Partial Dependence Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:

            feature = st.text_input("Enter the feature name for Partial Dependence Plot:")
            # Check if the input feature exists in the DataFrame
            if feature in train.columns:
                # Calculate partial dependence and plot
                try:
                    pdp = PartialDependenceDisplay.from_estimator(model, features=[feature], X=train)
                    fig = pdp.figure_
                    ax = fig.axes[0]
                    ax.set_title(f'Partial Dependence Plot: {feature}')
                    st.pyplot(fig)
                except (TypeError, ValueError) as e:
                    st.markdown(f"**Warning:** Error in Partial Dependence Plot for feature {feature}: {str(e)}")
            else:
                st.markdown(f"**Warning:** The feature '{feature}' is not in the DataFrame columns.")

        elif selected_plot == 'Learning Curve Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            # Calculate learning curves
            try:
                train_sizes, train_scores, test_scores = learning_curve(model, train, train_target, cv=5)
                
                # Plot learning curve
                fig, ax = plt.subplots()
                ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Loss', color='blue')
                ax.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Loss', color='red')
                ax.set_title('Learning Curve Plot')
                ax.set_xlabel('Number of Iterations')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** Error in Learning Curve Plot: {str(e)}")

# Define the sidebar for navigation
with st.sidebar:
    st.image(
        "https://www.atulhost.com/wp-content/uploads/2018/10/machine-learning-1536x1024.jpg"
    )
    st.title("AutoMLab")
    st.info("AutoMLab is an application designed to streamline the creation of automated machine learning workflows.")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Data Processing", "MLab"])

# Upload section
if choice == "Upload":
    st.title("Let's Get Started!")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_file = uploaded_file.name
            df.to_csv("sourcedata.csv", index=False)
            st.success("Data uploaded successfully!")
        except Exception as e:
            st.markdown(f"**Warning:** {str(e)}")

# Profiling section
elif choice == "Profiling":
    st.title("Automated EDA")
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
        missing_value_option = st.selectbox(
            "Choose how to handle missing values", ["Mean", "Median", "Mode", "Drop rows"]
        )
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
elif choice == "MLab":
    st.title("Machine Learning Lab")
    if os.path.exists("processed_data.csv"):
        df = pd.read_csv("processed_data.csv")

        # Display DataFrame columns for selecting the target variable
        st.write("DataFrame Columns:", df.columns)

        # Select target variable
        target_options = df.columns
        target = st.selectbox("Select Target Variable", target_options)

        # Validate selected target variable
        if not validate_target_variable(df, target):
            st.markdown(f"**Warning:** The selected target variable '{target}' is not in the DataFrame columns.")
        else:
            # Determine the analysis type (classification or regression)
            if df[target].dtype == "object" or len(df[target].unique()) < 10:
                problem = 'Classification'
            else:
                problem = 'Regression'

            st.write(f'Your target variable indicates a {problem} problem.')

            # Ask the user to choose a method
            if problem == 'Regression':
                method_options = list(REGRESSION_MODELS)
            else:
                method_options = list(CLASSIFICATION_MODELS)
                
            method = st.selectbox('Choose a Method', method_options)


            display_chart(df, target, method)
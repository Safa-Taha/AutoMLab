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


REGRESSION_MODELS = {
    'LinearRegression',
    'Ridge',
    'SVR',
    'DecisionTreeRegressor',
    'RandomForestRegressor',
    'KNeighborsRegressor',
    'GradientBoostingRegressor',
}


CLASSIFICATION_MODELS = {
    'LogisticRegression',
    'SVC',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
    'KNeighborsClassifier',
    'GradientBoostingClassifier',
}

def validate_target(df, target):
    return target in df.columns

def display(df, target, method):

    # Splitting the data into train and test sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_target = train[target]
    test_target = test[target]
    train = train.drop(target, axis=1)
    test = test.drop(target, axis=1)

    FeatureNames = train.columns

    model = MODELS[method]()
    model.fit(train, train_target)
    pred = model.predict(test)

    # Verifying data dimensions
    try:
        check_consistent_length(test_target, pred)
    except ValueError as e:
        st.markdown(f"**Warning:** {str(e)}")
        return

    # Displaying metrics (based on selected method)
    if method in REGRESSION_MODELS:
        mse = mean_squared_error(test_target, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_target, pred)

        st.write("Metrics:")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse}")
        st.write(f"R-Squared: {r2}")

        # Plot options for the user to choose from
        plot_choice = ['Actual vs Predicted Scatter Plot', 'Residual Plot']
        if method in ['LinearRegression', 'Ridge']:
            plot_choice.append('Coefficient Plot')
        if method == 'Ridge':
            plot_choice.append('Alpha vs Coefficient Plot')

        chosen_plot = st.selectbox('Choose a Plot to Display:', plot_choice)

        # Displaying the selected plot
        if chosen_plot == 'Actual vs Predicted Scatter Plot':
            fig, ax = plt.subplots()
            ax.scatter(test_target, pred, color='blue')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'Scatter Plot: {method}')
            st.pyplot(fig)

        elif chosen_plot == 'Residual Plot':
            residuals = test_target - pred
            fig, ax = plt.subplots()
            ax.scatter(pred, residuals, color='blue')
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Residuals')
            ax.set_title(f'Residual Plot: {method}')
            st.pyplot(fig)

        elif chosen_plot == 'Coefficient Plot' and method in ['LinearRegression', 'Ridge']:
            try:
                coefficients = model.coef_
                
                # Handling scalar values
                if isinstance(coefficients, np.float64):
                    st.markdown("**Warning:** Coefficients not available for the selected model.")
                    return
                
                # Ensuring 'coefficients' is an array & lengths of 'FeatureNames' and 'coefficients' match
                if len(coefficients[0]) != len(FeatureNames):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                fig, ax = plt.subplots()
                ax.bar(FeatureNames, coefficients[0], color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.set_title('Coefficient Plot')
                plt.xticks(rotation=45)  
                st.pyplot(fig)
            except (ValueError, AttributeError, TypeError) as e:
                st.markdown(f"**Warning:** {str(e)}")

        elif chosen_plot == 'Alpha vs Coefficient Plot' and method == 'Ridge':

            # Handling alphas and coefficients arrays
            alphas = np.logspace(-3, 3, 7)
            coeficients = []
            try:
                for alpha in alphas:
                    ridge_model = Ridge(alpha=alpha)
                    ridge_model.fit(train, train_target)
                    coeficients.append(ridge_model.coef_)

                if len(coeficients) > 0 and len(coeficients[0]) != len(FeatureNames):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                # Alpha vs coefficient
                fig, ax = plt.subplots()
                for coef, alpha in zip(coeficients, alphas):
                    ax.plot(FeatureNames, coef[0], label=f'Alpha: {alpha}')
                ax.set_title('Alpha vs Coefficient Plot')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.legend()
                plt.xticks(rotation=45) 
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** {str(e)}")

    elif method in CLASSIFICATION_MODELS:
        accuracy = accuracy_score(test_target, pred)
        precision = precision_score(test_target, pred, average='macro')
        recall = recall_score(test_target, pred, average='macro')
        f1 = f1_score(test_target, pred, average='macro')

        st.write("Metrics:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precision: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1-Score: {f1}")

        confusion_matrix_plotted = False
        
        # Plot options
        plot_choice = ['ROC Curve', 'Confusion Matrix']
        if method in ['LogisticRegression', 'SVC']:
            plot_choice.append('Coefficient Plot')
        if method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            plot_choice.append('Feature Importance Plot')
            plot_choice.append('Partial Dependence Plot')
            plot_choice.append('Learning Curve Plot')

        chosen_plot = st.selectbox('Choose a Plot to Display:', plot_choice)


        if chosen_plot == 'ROC Curve':
            ''' Ensuring the classification problem is binary '''
            if len(np.unique(test_target)) != 2:
                st.markdown("**Warning:** ROC Curves can only be plotted for binary classification problems.")
                return
            try:
                if method == 'LogisticRegression':
                    pred_prob = model.predict_proba(test)[:, 1]
                elif method == 'SVC':
                    decision_func = model.decision_func(test)
                    pred_prob = decision_func
                
                fpr, tpr, _ = roc_curve(test_target, pred_prob)
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

        elif chosen_plot == 'Confusion Matrix' and not confusion_matrix_plotted:
            cm = confusion_matrix(test_target, pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', ax=ax)
            ax.set_title(f'Confusion Matrix: {method}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)
            confusion_matrix_plotted = True

        elif chosen_plot == 'Coefficient Plot' and method in ['LogisticRegression', 'SVC']:
            try:
                coefficients = model.coef_
                
                # Handling scalar values
                if isinstance(coefficients, np.float64):
                    st.markdown("**Warning:** Coefficients not available for the selected model.")
                    return
                
                # Checking if lengths of FeatureNames and coefficients match
                if len(coefficients[0]) != len(FeatureNames):
                    st.markdown("**Warning:** Length mismatch between features and coefficients.")
                    return
                
                fig, ax = plt.subplots()
                ax.bar(FeatureNames, coefficients[0], color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Coefficient')
                ax.set_title('Coefficient Plot')
                plt.xticks(rotation=45) 
                st.pyplot(fig)
            except (ValueError, AttributeError, TypeError) as e:
                st.markdown(f"**Warning:** {str(e)}")

        elif chosen_plot == 'Feature Importance Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            try:
                feature_importances = model.feature_importances_
                
                fig, ax = plt.subplots()
                ax.bar(FeatureNames, feature_importances, color='blue')
                ax.set_xlabel('Features')
                ax.set_ylabel('Importance')
                ax.set_title('Feature Importance Plot')
                plt.xticks(rotation=45)  
                st.pyplot(fig)
            except Exception as e:
                st.markdown(f"**Warning:** Error in Feature Importance Plot: {str(e)}")

        elif chosen_plot == 'Partial Dependence Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:

            feature = st.text_input("Enter the feature name for Partial Dependence Plot:")
            if feature in train.columns:
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

        elif chosen_plot == 'Learning Curve Plot' and method in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            try:
                train_sizes, train_scores, test_scores = learning_curve(model, train, train_target, cv=5)
                
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

# Sidebar for navigation
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
    file_uploading = st.file_uploader("Upload a CSV file", type=["csv"])
    if file_uploading:
        try:
            df = pd.read_csv(file_uploading)
            st.session_state.file_uploading = file_uploading.name
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

        # Drop columns
        drop_columns = st.multiselect("Select columns to drop", df.columns)
        if drop_columns:
            df = df.drop(drop_columns, axis=1)

        # Handling missing values in NUMERICAL COLUMNS
        missing_values = st.selectbox(
            "Choose how to handle missing values", ["Mean", "Median", "Mode", "Drop rows"]
        )
        if missing_values in ["Mean", "Median", "Mode"]:
            numeric_columns = df.select_dtypes(include='number').columns
            for col in numeric_columns:
                if missing_values == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif missing_values == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif missing_values == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif missing_values == "Drop rows":
            df.dropna(inplace=True)

        # Encoding Categorical data
        encoding = st.selectbox("Choose how to encode categorical data", ["Label Encoding", "One-Hot Encoding"])
        if encoding == "Label Encoding":
            le = LabelEncoder()
            for col in df.select_dtypes(include='object').columns:
                df[col] = le.fit_transform(df[col])
        elif encoding == "One-Hot Encoding":
            df = pd.get_dummies(df)

        df.to_csv("processed_data.csv", index=False)
        st.success("Data processed and saved successfully!")

# MLab section
elif choice == "MLab":
    st.title("Machine Learning Lab")
    if os.path.exists("processed_data.csv"):
        df = pd.read_csv("processed_data.csv")

        st.write("DataFrame Columns:", df.columns)

        # Selecting and validating the target variable
        target_options = df.columns
        target = st.selectbox("Select Target Variable", target_options)
        if not validate_target(df, target):
            st.markdown(f"**Warning:** The selected target variable '{target}' is not in the DataFrame columns.")
        
        else:
            # Determining the analysis type (Classification or Regression) 
            if df[target].dtype == "object" or len(df[target].unique()) < 10:
                problem = 'Classification'
            else:
                problem = 'Regression'
            st.write(f'Your target variable indicates a {problem} problem.')

            if problem == 'Regression':
                method_choice = list(REGRESSION_MODELS)
            else:
                method_choice = list(CLASSIFICATION_MODELS)
                
            method = st.selectbox('Choose a Method', method_choice)


            display(df, target, method)
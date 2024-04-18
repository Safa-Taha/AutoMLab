import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import learning_curve, ShuffleSplit
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

best_model = None

# Define the EDA class
class EDA:
    def __init__(self, df):
        self.df = df

    def cat_feat(self):
        df_cat = self.df.select_dtypes(include='object')
        return df_cat.columns

    def cat_nulls(self):
        df_cat = self.df.select_dtypes(include='object')
        return df_cat.columns[df_cat.isna().sum() > 0]

    def nums_nulls(self):
        df_num = self.df.select_dtypes(include=np.number)
        return df_num.columns[df_num.isna().sum() > 0]

    def fill_with_mean(self, col):
        imputer = SimpleImputer(strategy='mean')
        self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))
        return self.df

    def fill_with_median(self, col):
        imputer = SimpleImputer(strategy='median')
        self.df[col] = imputer.fit_transform(self.df[col].values.reshape(-1, 1))
        return self.df

    def fill_with_mode(self, col):
        most_frequent_value = self.df[col].mode()[0]
        self.df[col].fillna(most_frequent_value, inplace=True)
        return self.df

    def fill_with_constant(self, col, value):
        self.df[col].fillna(value, inplace=True)
        return self.df

    def one_hot_encoder(self, col):
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_data = encoder.fit_transform(self.df[col].values.reshape(-1, 1))
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names([col]))
        self.df = pd.concat([self.df, encoded_df], axis=1).drop(col, axis=1)
        return self.df

    def label_encoder(self, col):
        encoder = LabelEncoder()
        self.df[col] = encoder.fit_transform(self.df[col])
        return self.df

    def encoding(self):
        self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df

# Define the plotting class
class Plotting:
    def __init__(self):
        pass

    def plot_confusion_matrix(self, confusion_matrix, labels):
        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(confusion_matrix.shape[1]),
               yticks=np.arange(confusion_matrix.shape[0]),
               xticklabels=labels, yticklabels=labels,
               ylabel='True label',
               xlabel='Predicted label',
               title='Confusion Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
        plt.show()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        plt.show()

    def plot_learning_curve(self, estimator, title, X, y, scoring):
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel("Training examples")
        ax.set_ylabel(scoring)
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
            n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), scoring=scoring
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
        ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
        ax.legend(loc="best")
        plt.show()

    def feature_importance_plot(self, feature_importances, feature_names):
        feat_imp_series = pd.Series(feature_importances, index=feature_names)
        sorted_feat_imp = feat_imp_series.sort_values(ascending=False)
        sorted_feat_imp.plot(kind='bar', color='blue')
        plt.xlabel('Feature Names')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()

# Function to load data and perform EDA
def load_and_process_data(file):
    # Load data using pandas
    data = pd.read_csv(file)

    # Initialize EDA instance with the data
    eda = EDA(data)

    # Display categorical features and columns with missing values
    st.write("Categorical features:", eda.cat_feat())
    st.write("Categorical columns with missing values:", eda.cat_nulls())
    st.write("Numerical columns with missing values:", eda.nums_nulls())

    # Options for filling missing values
    if st.button("Fill missing values with mean"):
        for col in eda.nums_nulls():
            eda.fill_with_mean(col)
        st.write("Missing values filled with mean.")
    if st.button("Fill missing values with median"):
        for col in eda.nums_nulls():
            eda.fill_with_median(col)
        st.write("Missing values filled with median.")
    if st.button("Fill missing values with mode"):
        for col in eda.cat_nulls():
            eda.fill_with_mode(col)
        st.write("Missing values filled with mode.")
    if st.button("Fill missing values with constant"):
        value = st.text_input("Enter constant value:")
        if value:
            for col in eda.cat_nulls():
                eda.fill_with_constant(col, value)
            st.write("Missing values filled with constant value.")

    # Encoding options
    if st.button("One-hot encode categorical columns"):
        for col in eda.cat_feat():
            eda.one_hot_encoder(col)
        st.write("One-hot encoded categorical columns.")
    if st.button("Label encode categorical columns"):
        for col in eda.cat_feat():
            eda.label_encoder(col)
        st.write("Label encoded categorical columns.")
    return eda.df

# Function to visualize data using the Plotting class
def visualize_data(df, column1, column2, plot_type):
    plotter = Plotting()

    if plot_type == 'histogram':
        plt.hist(df[column1], bins=20, alpha=0.7, color='blue', label=column1)
        plt.hist(df[column2], bins=20, alpha=0.7, color='orange', label=column2)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column1} and {column2}')
        plt.legend()
        plt.show()
        
    elif plot_type == 'boxplot':
        sns.boxplot(data=df[[column1, column2]])
        plt.title(f'Box Plot of {column1} and {column2}')
        plt.show()
        
    elif plot_type == 'scatter':
        sns.scatterplot(x=df[column1], y=df[column2])
        plt.title(f'Scatter Plot of {column1} vs {column2}')
        plt.show()
        
    elif plot_type == 'heatmap':
        heatmap_data = df.pivot_table(index=column1, columns=column2, aggfunc='size')
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True)
        plt.title(f'Heatmap of {column1} and {column2}')
        plt.show()

    elif plot_type == 'pairplot':
        sns.pairplot(df[[column1, column2]], diag_kind='kde')
        plt.title(f'Pairplot of {column1} and {column2}')
        plt.show()

    elif plot_type == 'lineplot':
        sns.lineplot(x=df[column1], y=df[column2])
        plt.title(f'Line Plot of {column1} and {column2}')
        plt.show()

    elif plot_type == 'violin':
        sns.violinplot(data=df, x=column1, y=column2)
        plt.title(f'Violin Plot of {column1} and {column2}')
        plt.show()

    elif plot_type == 'correlation':
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
        plt.title('Correlation Matrix')
        plt.show()

    else:
        st.write('Invalid plot type selected.')

# Main function
def main():
    st.title('Data Analysis and Visualization App')
    
    # Load data
    file = st.file_uploader("Upload a dataset", type=["csv"])
    if file is not None:
        df = load_and_process_data(file)
        st.write("Data loaded successfully.")
        
        # Visualization options
        st.subheader('Data Visualization')
        plot_type = st.selectbox("Select plot type", ['histogram', 'boxplot', 'scatter', 'heatmap', 'pairplot', 'lineplot', 'violin', 'correlation'])
        column1 = st.selectbox('Select first column', df.columns)
        column2 = st.selectbox('Select second column', df.columns)
        
        # Visualize data
        visualize_data(df, column1, column2, plot_type)
    
    # Model selection and evaluation
    st.subheader('Model Selection and Evaluation')
    target_column = st.selectbox('Select target column', df.columns)
    problem_type = st.radio('Problem type', ('Classification', 'Regression'))
    
    if st.button('Run PyCaret'):
        if problem_type == 'Classification':
            clf_exp = ClassificationExperiment()
            clf_exp.setup(data=df, target=target_column)
            best_model = clf_exp.compare_models()
            st.write(f"Best model: {best_model}")
        elif problem_type == 'Regression':
            reg_exp = RegressionExperiment()
            reg_exp.setup(data=df, target=target_column)
            best_model = reg_exp.compare_models()
            st.write(f"Best model: {best_model}")

        # Plotting confusion matrix or feature importance
        plotter = Plotting()
        if problem_type == 'Classification':
            if st.button('Plot confusion matrix'):
                cm = clf_exp.plot_model(best_model, plot='confusion_matrix')
                plotter.plot_confusion_matrix(cm, labels=clf_exp.get_config('data').columns)
        elif problem_type == 'Regression':
            if st.button('Plot feature importance'):
                feat_importance = reg_exp.plot_model(best_model, plot='feature')
                plotter.feature_importance_plot(feat_importance, reg_exp.get_config('data').columns)

if __name__ == '__main__':
    main()
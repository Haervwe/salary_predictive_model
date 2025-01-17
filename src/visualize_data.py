import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def visualize_dataset(df:pd.DataFrame)->None:
    # List of numerical columns
    numerical_cols = ['Age', 'Years of Experience', 'Salary']

    # Create box plots for each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot of {col}')
        plt.savefig(f'plots/boxplot_{col.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        
    # Create histograms for each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'plots/histogram_{col.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        
    # Pairs of variables to plot
    pairs = [('Age', 'Salary'), ('Years of Experience', 'Salary'), ('Age', 'Years of Experience')]

    for x_col, y_col in pairs:
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_col], df[y_col])
        plt.title(f'Scatter Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig(f'plots/scatter_{y_col.replace(" ", "_")}_vs_{x_col.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        
    #categorical columns
    categorical_cols = ['Gender', 'Education Level']

    for col in categorical_cols:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=col)
        plt.title(f'Count Plot of {col}')
        plt.savefig(f'plots/countplot_{col.replace(" ", "_")}.png', bbox_inches='tight')
        plt.close()
        
    # Visualize top 10 job titles
    top_n = 10
    top_job_titles = df['Job Title'].value_counts().nlargest(top_n).index
    filtered_dataset = df[df['Job Title'].isin(top_job_titles)]

    plt.figure(figsize=(12, 6))
    sns.countplot(data=filtered_dataset, y='Job Title', order=top_job_titles)
    plt.title('Top 10 Job Titles')
    plt.savefig('plots/top_10_job_titles.png', bbox_inches='tight')
    plt.close()


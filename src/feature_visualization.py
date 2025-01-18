import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_relationships(x_train: pd.DataFrame, y_train: pd.Series):
    
    train_data = pd.concat([x_train, y_train], axis=1)
    """
    Plots the relationships between features in the dataset, focusing on how well job_title_encoded correlates with salary.
    
    Parameters:
    data (pd.DataFrame): The dataset containing the features and target variable.
    """

    # Plot the relationship between job_title_encoded and salary
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Job Title Encoded', y='Salary', data=train_data)
    plt.title('Job Title Encoded vs. Salary')
    plt.xlabel('Job Title Encoded')
    plt.ylabel('Salary')
    plt.savefig('plots/job_title_encoded_vs_salary.png', bbox_inches='tight')
    plt.close()
    
    # Pairplot
    sns.pairplot(train_data)
    plt.savefig('plots/pairplot.png', bbox_inches='tight')
    print("Pairplot saved to 'plots/pairplot.png'")
    plt.close()

    # Correlation heatmap
    corr_matrix = train_data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.savefig('plots/correlation_heatmap.png', bbox_inches='tight')
    print("Correlation heatmap saved to 'plots/correlation_heatmap.png'")
    plt.close()

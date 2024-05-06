import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def feature_distribution_visualization(df):
    # Get the list of features
    features = df.columns
    
    # Set up the figure
    num_features = len(features)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols
    plt.figure(figsize=(9, 3*num_rows))
    
    # Plot each feature
    '''for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i+1)
        if df[feature].dtype == 'object':
            sns.countplot(x=feature, data=df, palette='Set2')
        else:
            plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
        plt.title(feature)
        plt.xlabel('Value')
        plt.ylabel('Frequency')'''
    for i, feature in enumerate(features):
        plt.subplot(num_rows, num_cols, i+1)
        if df[feature].dtype == 'object':
            # For categorical variables, use countplot
            sns.countplot(x=feature, data=df, palette='Set2')
            plt.title(feature)
        else:
            # For numerical variables, use histogram with logarithmic scale
            nonzero_values = df[feature][df[feature] != 0]  # Exclude zero values for log scale
            log_bins = np.geomspace(np.min(nonzero_values), np.max(df[feature]), num=20)
            plt.hist(df[feature], bins=log_bins, color='skyblue', edgecolor='black')
            plt.xscale('log')  # Set x-axis to log scale
            plt.title(feature)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def feature_correlation_visualization(df):
    # Calculate the correlation matrix
    drop_table = []
    for i in df.columns:
        if df[i].dtype == 'object':
            drop_table.append(i)
    for i in drop_table:
        df = df.drop(columns=i)

    corr_matrix = df.corr()
    
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    
    plt.title('Feature Correlation Matrix')
    plt.show()

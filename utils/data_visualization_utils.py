import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd

def feature_distribution_visualization(df):
    # Get the list of features
    features = df.columns
    new_features = []
    for i in features:
        if df[i].dtype == 'object':
            new_features.append(i)
    features = new_features
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
            ax = sns.countplot(x=feature, data=df, palette='Set2')
            if feature == "date":
                ax.set_xticklabels(ax.get_xticklabels(), visible=False)
            if feature == "location":
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45, ha="right")
            plt.title(feature)
        '''else:
            # For numerical variables, use histogram with logarithmic scale
            nonzero_values = df[feature][df[feature] != 0]  # Exclude zero values for log scale
            #log_bins = np.geomspace(np.min(nonzero_values), np.max(df[feature]), num=20)
            plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
            #plt.xscale('log')  # Set x-axis to log scale
            plt.title(feature)
            plt.xlabel('Value')
            plt.ylabel('Frequency')'''
    
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

def PCA_analysis(df):
    # Select only numerical columns for PCA
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Standardize the numerical data
    standardized_data = (numerical_df - numerical_df.mean()) / numerical_df.std()
    
    # Perform PCA
    pca = PCA()
    principal_components = pca.fit_transform(standardized_data)
    
    # Explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Plot explained variance ratio
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(explained_variance_ratio), marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.grid(True)
    
    # Plot key component plot

    class_colors = {'N': 'red', 'C': 'blue', 'T': 'green', 'M': 'orange'}  # Add more if needed
    #class_colors = {'B': 'red', 'C': 'blue', 'T': 'green', 'M': 'orange'}  # Add more if needed

    
    # Create a custom colormap using ListedColormap
    plt.subplot(1, 2, 2)
    for category, color in class_colors.items():
        plt.scatter(principal_components[df["class"] == category, 0], 
                    principal_components[df["class"] == category, 1], 
                    color=color, 
                    label=category)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Key Component Plot')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.tight_layout()
    plt.show()
    
    # Return the PCA results
    result_dict =  {
        'principal_components': principal_components,
        'explained_variance_ratio': explained_variance_ratio,
        'pca': pca
    }

    loadings = pca.components_[:4]
    loading_matrix = pd.DataFrame(loadings.T, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=numerical_df.columns)
    print(loading_matrix)

    return result_dict


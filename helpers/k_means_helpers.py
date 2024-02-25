# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans


def compute_kmeans_cluster_values(n_clusters, df):
    """
    Method to initialize, fit, and predict using KMeans.

    Parameters:
        n_clusters (int): An int parameter for the number of KMeans clusters.
        df (dataframe): A dataframe parameter for KMeans predictions.

    Returns:
        ndarray: A ndarray of cluster values.
    """

    # Initialize the K-Means model using the best value for k
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=1)

    # Fit the K-Means model using the data
    model.fit(df)

    # Predict the clusters to group the cryptocurrencies using the data
    k = model.predict(df)

    # Return resulting array of cluster values
    return k


def create_inertia_df(df, n_clusters=10):
    """
    Method to compute k and inertia values from a dataframe.
    Creates a new dataframe with the k and inertia values to plot an Elbow curve.

    Parameters:
        df (dataframe): A dataframe used in KMeans model.
        n_clusters (int, optional): An int parameter for the number of KMeans clusters. Default is 10.

    Returns:
        tuple: A new dataframe with k and inertia values.
    """

    # Create a list with the number of k-values to try
    # Use a range from 1 to n_clusters + 1
    # Default range is 1 to 11 to create 10 clusters
    k = list(range(1, n_clusters + 1))

    # Create an empty list to store the inertia values
    inertia = []

    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using the dataFrame
    # 3. Append the model.inertia_ to the inertia list
    for i in k:
        k_model = KMeans(n_clusters=i, n_init="auto", random_state=1)
        k_model.fit(df)
        inertia.append(k_model.inertia_)

    # Create a dictionary with the data to plot the Elbow curve
    elbow_data = {"k": k, "inertia": inertia}

    # Create a DataFrame with the data to plot the Elbow curve
    df_elbow = pd.DataFrame(elbow_data)

    # Return a dataframe and a list of k-values
    return df_elbow, k

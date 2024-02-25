# Import required libraries and dependencies
from sklearn.cluster import KMeans


def compute_kmeans(n_clusters, df):
    """
    Method to initialize, fit, and predict using KMeans.

    Parameters:
        n_clusters (int): An int parameter for the number of KMeans clusters.
        df (dataframe): A dataframe parameter for KMeans predictions.

    Returns:
        ndarray: A ndarray of cluster values.
    """

    # Initialize the K-Means model using the best value for k
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0)

    # Fit the K-Means model using the data
    model.fit(df)

    # Predict the clusters to group the cryptocurrencies using the data
    k = model.predict(df)

    # Return resulting array of cluster values
    return k

# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans


def compute_inertia_df(df):
    """
    Method to compute k and inertia values from a dataframe.
    Creates a new dataframe with the k and inertia values to plot an Elbow curve.

    Parameters:
        df (dataframe): A dataframe used in KMeans model.

    Returns:
        tuple: A dataframe and k result.
    """

    # Create a list with the number of k-values to try
    # Use a range from 1 to 11
    k = list(range(1, 11))

    # Create an empty list to store the inertia values
    inertia = []

    # Create a for loop to compute the inertia with each possible value of k
    # Inside the loop:
    # 1. Create a KMeans model using the loop counter for the n_clusters
    # 2. Fit the model to the data using the dataFrame
    # 3. Append the model.inertia_ to the inertia list
    for i in k:
        k_model = KMeans(n_clusters=i, n_init="auto", random_state=0)
        k_model.fit(df)
        inertia.append(k_model.inertia_)

    # Create a dictionary with the data to plot the Elbow curve
    elbow_data = {"k": k, "inertia": inertia}

    # Create a DataFrame with the data to plot the Elbow curve
    df_elbow = pd.DataFrame(elbow_data)

    # Return a dataframe and a list of integers
    return df_elbow, k

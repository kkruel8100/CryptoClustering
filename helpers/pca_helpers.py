# Import required libraries and dependencies
from sklearn.decomposition import PCA


def create_pca_df(n_components, df):
    """
    Method to pca model and transform from a dataframe.
    Creates a new dataframe with the principal components and pca model.

    Parameters:
        n_components (int): An int parameter for the number of PCA components.
        df (dataframe): A dataframe used in PCA model.

    Returns:
        tuple: A new dataframe with transformed data and pca model.
    """

    # Create a PCA model instance and set `n_components`.
    pca = PCA(n_components=n_components)

    # Use the PCA model with `fit_transform` on the original DataFrame to reduce to n_components principal components.
    market_pca = pca.fit_transform(df)

    # Return the resulting DataFrame and pca model
    return market_pca, pca

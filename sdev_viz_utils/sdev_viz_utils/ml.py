# 

def pd_scale_norm_df(df):
    """
    * type-def ::(pd.DataFrame) -> pd.DataFrame
    * ---------------{Function}---------------
        * Scales and normalizes a DataFrame by applying StandardScaler to numerical columns.
    * ----------------{Returns}---------------
        * : pd.DataFrame | The scaled and normalized DataFrame
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The DataFrame to scale and normalize
    * ----------------{Usage}-----------------
        * >>> scaled_df = pd_scale_norm_df(df)
    * ----------------{Notes}-----------------
        * This function is useful when preprocessing data for machine learning models, as many models work better with normalized features.
    """
    import sklearn as sk

    dtype_cols = [i for i in df.columns if df[i].dtype == np.object]
    cols_to_norm = df.loc[:, ~df.columns.isin(dtype_cols)].columns
    train = df
    train[cols_to_norm] = sk.preprocessing.StandardScaler().fit_transform(
        train[cols_to_norm]
    )
    return train[cols_to_norm]


def histogram_intersection(a, b):
    """
    * type-def ::(np.array, np.array) -> float
    * ---------------{Function}---------------
        * Computes the histogram intersection between two arrays.
    * ----------------{Returns}---------------
        * : float | The histogram intersection value
    * ----------------{Params}----------------
        * : a ::np.array | The first array
        * : b ::np.array | The second array
    * ----------------{Usage}-----------------
        * >>> intersection_value = histogram_intersection(a, b)
    * ----------------{Notes}-----------------
        * Histogram intersection can be used as a similarity metric between two histograms, indicating the degree of overlap between the histograms.
    """
    v = np.minimum(a, b).sum().round(decimals=1)
    return v


def plot_collinearity(df, return_corr=False):
    """
    * type-def ::(pd.DataFrame) -> None
    * ---------------{Function}---------------
        * Plots a heatmap of collinearity between the columns of a DataFrame using the Pearson correlation coefficient.
    * ----------------{Returns}---------------
        * : None
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The DataFrame to analyze for collinearity
    * ----------------{Usage}-----------------
        * >>> plot_collinearity(df)
    * ----------------{Notes}-----------------
        * This function can be useful for visualizing the relationships between features in a dataset to identify collinearity.
    """
    import sklearn as sk
    import seaborn as sns

    dtype_cols = [i for i in df.columns if df[i].dtype == np.object]
    cols_to_norm = df.loc[:, ~df.columns.isin(dtype_cols)].columns
    train = df.copy()
    train[cols_to_norm] = sk.preprocessing.StandardScaler().fit_transform(
        train[cols_to_norm]
    )

    # Identify collinearity between columns¶
    corr_df = train[cols_to_norm].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corr_df)

    
    if return_corr:
        return corr_df
    else:
        plt.show()
        return None



def learned_frontier(data, classifier, X_train, X_test, savefig=None):
    """
    * type-def ::(np.array, sklearn classifier, np.array, np.array, Optional[str]) -> None
    * ---------------{Function}---------------
        * Plots the learned frontier of a classifier on the given data.
    * ----------------{Returns}---------------
        * : None
    * ----------------{Params}----------------
        * : data ::np.array | The dataset used for plotting
        * : classifier ::sklearn classifier | The classifier to fit and plot
        * : X_train ::np.array | The training data
        * : X_test ::np.array | The testing data
        * : savefig ::Optional[str] | The path to save the generated plot; if None, the plot will not be saved
    * ----------------{Usage}-----------------
        * >>> learned_frontier(data, classifier, X_train, X_test)
    * ----------------{Notes}-----------------
        * This function is useful for visualizing the decision boundary or learned frontier of a classifier on a given dataset.
    """
    import matplotlib.pyplot as plt

    data_min = multi_dim_min(data)
    data_min = data_min + data_min / 2
    data_max = multi_dim_max(data)
    data_max = data_max + data_max / 2

    # fit the model for novelty detection
    clf = classifier
    clf.fit(X_train)

    y_pred_test = clf.predict(X_test)

    # xx, yy = np.meshgrid(np.linspace(-30, 30, 5000), np.linspace(-30, 30, 5000))
    xx, yy = np.meshgrid(
        np.linspace(data_min, data_max, 5000), np.linspace(data_min, data_max, 5000)
    )

    print(f"Learning frontier for {type(clf)}")
    # plot the learned frontier, the points, and the nearest vectors to the plane
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title(f"Learning frontier with  {type(clf)}")
    plt.contourf(
        xx, yy, Z, levels=np.linspace(Z.min() + Z.min() / 4, 10, 7), cmap=plt.cm.PuBu
    )

    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
    plt.contourf(xx, yy, Z, levels=[0, Z.max() + Z.max() / 4], colors="palevioletred")

    s = 10
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")

    plt.axis("tight")
    plt.xlim((data_min, data_max))
    plt.ylim((data_min, data_max))
    plt.legend(
        [a.collections[0], b1, b2],
        [
            "learned frontier",
            "training observations",
            "new regular_observations",
        ],
        loc="upper left",
        prop=matplotlib.font_manager.FontProperties(size=11),
    )

    n_error_test = y_pred_test[y_pred_test == -1].size

    # plt.xlabel("errors novel regular: %d/40 ;" % (n_error_test))
    plt.show()

    if savefig != None:
        plt.savefig(savefig)
    return None


def plot_colinearity_variations(df):
    """
    * type-def ::(pd.DataFrame) -> matplotlib.figure.Figure
    * ---------------{Function}---------------
        * Plots the collinearity between columns in the dataframe using various similarity measures.
    * ----------------{Returns}---------------
        * : fig ::matplotlib.figure.Figure | The resulting plot figure.
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The dataframe containing the data to be analyzed.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> fig = plot_colinearity_variations(df)
    * ----------------{Notes}-----------------
        * This function is useful for visualizing the collinearity between columns in a dataset using different similarity
          * measures. The function generates a heatmap for each similarity measure and arranges them in a grid layout for easy
          * comparison.
    * ----------------{Side Effects}---------
        * This function may generate a large figure with multiple heatmaps, which can consume a significant amount of memory
          * depending on the size of the input DataFrame. Be mindful of the available system resources when using this function
          * with large datasets.
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame. However, it creates a new DataFrame `train_df` based on the input
          * DataFrame, which is then used to compute correlations. The original DataFrame remains unchanged.
    """
    from pypair.association import binary_binary

    
    # Similarity measure definitions
    jaccard = lambda a, b: binary_binary(a, b, measure="jaccard")
    tanimoto = lambda a, b: binary_binary(a, b, measure="tanimoto_i")
    # This measure is typically used to judge the similarity between two clusters.
    ochiai = lambda a, b: binary_binary(a, b, measure="ochia_i")
    # Yule's Q is based off of the odds ratio or cross-product ratio, a measure of proportional reduction in error (PRE)
    yule = lambda a, b: binary_binary(a, b, measure="yule_q")
    #  A higher mutual information value implies strong association
    m_inf = lambda a, b: binary_binary(a, b, measure="mutual_information")
    # Tetrachoric correlation ranges from :math:`[-1, 1]`, where 0 indicates no agreement,
    # 1 indicates perfect agreement and -1 indicates perfect disagreement.
    tetrachoric = lambda a, b: binary_binary(a, b, measure="tetrachoric")

    train_df = pd_scale_norm_df(df)
    fig = plt.figure(figsize=(20, 15))
    
    measures = [
        ("pearson", "collinearity, Pearson similarity measure: "),
        ("spearman", "Spearman similarity measure: "),
        (histogram_intersection, "collinearity, Histogram Intersection similarity measure: "),
        (jaccard, "Jaccard similarity measure: "),
        (tanimoto, "Tanimoto similarity measure (Jaccard Index): "),
        (ochiai, "Ochiai similarity measure (cosine similarity): "),
        (yule, "yule Q measure (cosine similarity): "),
        (m_inf, "Mutual Information: "),
        (tetrachoric, "tetrachoric correlation: "),
    ]

    # Plotting the heatmaps for various similarity measures
    
    for i, (method, title) in enumerate(measures, 1):
        corr_df = train_df.corr(method=method)
        ax = fig.add_subplot(3, 3, i)
        ax.title.set_text(title)
        sns.heatmap(corr_df, ax=ax, cmap="RdYlBu")

    fig.tight_layout()
    return fig


def pd_visualize_cat_cols(df, col_name):
    if col_name in df.select_dtypes(include=['object']).columns:
        fig = px.bar(df[col_name].value_counts().reset_index(), x='index', y=col_name)
        fig.show()
    else:
        print("Column not found or not categorical")
    




def visualize_categoricals(df, filter_contains='', filter_regex='', columns=None, N=5, max_plots=16):
    
    import pandas as pd
    import plotly.express as px
    import math
    import re
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    # Example usage
    # visualize_categoricals(cb, filter_contains='cb', N=5, max_plots=3)
    object_columns = df.select_dtypes(include=['object']).columns

    if filter_contains:
        object_columns = [col for col in object_columns if filter_contains in col]

    if filter_regex:
        pattern = re.compile(filter_regex)
        object_columns = [col for col in object_columns if pattern.search(col)]

    if columns is not None:
        valid_columns = [col for col in columns if col in object_columns]
    else:
        valid_columns = object_columns

    # Take only the columns with non-zero top categories
    valid_columns = [col for col in valid_columns if df[col].value_counts().nlargest(N).sum() > 0]

    # Limiting to max_plots
    valid_columns = valid_columns[:max_plots]

    num_cols = min(4, len(valid_columns))
    num_rows = math.ceil(len(valid_columns) / num_cols)

    fig = make_subplots(rows=num_rows, cols=num_cols)
    for idx, col in enumerate(valid_columns):
        row = idx // num_cols + 1
        col_idx = idx % num_cols + 1
        top_categories = df[col].value_counts().nlargest(N)
        fig.add_trace(
            go.Bar(x=top_categories.index, y=top_categories.values, name=col),
            row=row, col=col_idx
        )

    fig.update_layout(height=300 * num_rows, width=400 * num_cols, title_text="Top N Categories")
    fig.show()

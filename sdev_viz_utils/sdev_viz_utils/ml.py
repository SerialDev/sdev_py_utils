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
        (
            histogram_intersection,
            "collinearity, Histogram Intersection similarity measure: ",
        ),
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
    """
    • ---------------Function---------------
    • Visualize a categorical column in a pandas DataFrame using a bar plot
    • ----------------Returns---------------
    • -> result ::str
    • Either 'Success' if the operation was successful, or 'Failure' otherwise
    • ----------------Params----------------
    • df ::DataFrame
    • The input pandas DataFrame
    • col_name ::str
    • The name of the column to visualize
    • ----------------Usage-----------------
    • pd_visualize_cat_cols(df, col_name)
    • Visualizes a categorical column in a pandas DataFrame using a bar plot.
    • The function first checks if the column name exists and is categorical.
    • If the column does not exist or is not categorical, the function will print
    • "Column not found or not categorical" and return 'Failure'.
    • If the column is categorical, the function will create a bar plot using
      plotly
    • express and display it. The plot will show the count of each unique value
    • in the column.
    •
    • Example:
    • .. code-block:: python
    •
    • import pandas as pd
    • import plotly.express as px
    • df = pd.DataFrame({' column1':['a','b','a','b','a','c']})
    • result = pd_visualize_cat_cols(df, 'column1')
    •
    • In this example, the function will visualize the categorical column
      'column1'
    • in the input DataFrame 'df' using a bar plot. The resulting plot will show
    • the count of each unique value ('a', 'b', and 'c') in the column.
    •
    • Returns 'Success' if the operation was successful, or 'Failure' otherwise.
    •
    • Note: This function uses plotly.express, which must be installed for the
    • function to work. To install plotly.express, run pip install plotly.
    •
    • This function will not modify the input DataFrame.
    """
    if col_name in df.select_dtypes(include=["object"]).columns:
        fig = px.bar(df[col_name].value_counts().reset_index(), x="index", y=col_name)
        fig.show()
    else:
        print("Column not found or not categorical")


def visualize_categoricals(
    df, filter_contains="", filter_regex="", columns=None, N=5, max_plots=16
):
    """
    * ----------------Function----------------
    * This function visualizes the top N categories of the categorical columns in a given pandas DataFrame.
    * ----------------Returns----------------
    * -> result ::str |'Success' if the operation was successful, 'Failure' otherwise
    * ----------------Params----------------
    * df :: DataFrame | The input pandas DataFrame.
    * filter_contains :: str, optional | The string to be contained in the column name for filtering. Defaults to ''.
    * filter_regex :: str, optional | The regular expression pattern to filter the column names. Defaults to ''.
    * columns :: list, optional | The list of column names to be visualized. If not provided, all categorical columns will be visualized. Defaults to None.
    * N :: int, optional | The number of top categories to be displayed for each column. Defaults to 5.
    * max_plots :: int, optional | The maximum number of plots to be created. Defaults to 16.
    * ----------------Usage-----------------
    *
    * visualize_categoricals(df, filter_contains='', filter_regex='', columns=None, N=5, max_plots=16)
    *
    *
    * The function visualizes the top N categories of the categorical columns in a
    * given pandas DataFrame. It can filter the columns based on a specified string
    * (filter_contains) or regular expression pattern (filter_regex). Users can also
    * specify the list of columns (columns) to be visualized. The function shows the
    * top N (default: 5) categories for each column in a subplot. A maximum of max
    * _plots (default: 16) plots can be created.
    """
    import pandas as pd
    import plotly.express as px
    import math
    import re
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Example usage
    # visualize_categoricals(cb, filter_contains='cb', N=5, max_plots=3)
    object_columns = df.select_dtypes(include=["object"]).columns

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
    valid_columns = [
        col for col in valid_columns if df[col].value_counts().nlargest(N).sum() > 0
    ]

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
            row=row,
            col=col_idx,
        )

    fig.update_layout(
        height=300 * num_rows, width=400 * num_cols, title_text="Top N Categories"
    )
    fig.show()


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


def plot_feature_importances(
    model,
    columns,
    prefixes_to_merge=[
        "embeddings_",
        "UMAP",
        "count",
    ],
    contains_to_merge=[],
    top_n=40,
    height=1000,
    output_filename="plot",
):
    feature_scores = pd.Series(model.feature_importances_, index=columns).sort_values(
        ascending=False
    )

    def merge_feature_prefixes(feature_scores, prefixes_to_merge, contains_to_merge):
        merged_scores = {}
        hover_info = {}
        feature_count = {}
        feature_medians = {}
        top_contributors = {}

        for substring in contains_to_merge:
            matching_features = feature_scores[
                feature_scores.index.str.contains(substring, case=False)
            ]
            if not matching_features.empty:
                aggregate_prefix = f"aggregate_contains_{substring}"
                merged_scores[aggregate_prefix] = matching_features.sum()
                hover_info[aggregate_prefix] = ", ".join(
                    matching_features.index[:5]
                ) + ("..." if len(matching_features) > 5 else "")
                feature_count[aggregate_prefix] = len(matching_features)
                feature_medians[aggregate_prefix] = matching_features.median()
                top_contributors[aggregate_prefix] = matching_features.idxmax()
                feature_scores = feature_scores[
                    ~feature_scores.index.str.contains(substring, case=False)
                ]
            else:
                print(f"No matching features found for substring '{substring}'")

        for prefix in prefixes_to_merge:
            matching_features = feature_scores[
                feature_scores.index.str.startswith(prefix)
            ]
            if not matching_features.empty:
                aggregate_prefix = f"aggregate_{prefix}"
                merged_scores[aggregate_prefix] = matching_features.sum()
                hover_info[aggregate_prefix] = ", ".join(
                    matching_features.index[:5]
                ) + ("..." if len(matching_features) > 5 else "")
                feature_count[aggregate_prefix] = len(matching_features)
                feature_medians[aggregate_prefix] = matching_features.median()
                top_contributors[aggregate_prefix] = matching_features.idxmax()
                feature_scores = feature_scores[
                    ~feature_scores.index.str.startswith(prefix)
                ]
            else:
                print(f"No matching features found for prefix '{prefix}'")

        merged_scores_series = pd.Series(merged_scores)
        feature_scores = pd.concat([feature_scores, merged_scores_series])
        return (
            feature_scores,
            hover_info,
            feature_count,
            feature_medians,
            top_contributors,
        )

    (
        merged_feature_scores,
        hover_info,
        feature_count,
        feature_medians,
        top_contributors,
    ) = merge_feature_prefixes(feature_scores, prefixes_to_merge, contains_to_merge)
    merged_feature_scores = merged_feature_scores.sort_values(ascending=False)

    merged_feature_scores = merged_feature_scores.head(top_n)

    feature_scores_df = merged_feature_scores.reset_index()
    feature_scores_df.columns = ["Feature", "Importance"]

    def truncate_label(label):
        return (label[:47] + "...") if len(label) > 50 else label

    feature_scores_df["Feature"] = feature_scores_df["Feature"].apply(
        lambda x: (
            truncate_label(f"{x}[{top_contributors[x]}]")
            + f" <span style='color:green;'>+{feature_count[x]}</span>"
            if x in hover_info
            else truncate_label(x)
        )
    )

    zebra_shapes = []
    for i in range(len(feature_scores_df)):
        zebra_shapes.append(
            dict(
                type="rect",
                x0=0,
                y0=i - 0.5,
                x1=feature_scores_df["Importance"].max() * 1.1,
                y1=i + 0.5,
                fillcolor="lightgray" if i % 2 == 0 else "white",
                opacity=0.1,
                layer="below",
                line_width=0,
            )
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=feature_scores_df["Importance"],
            y=feature_scores_df.index,
            mode="markers+lines",
            name="Importance",
            line=dict(color="rgba(100, 100, 100, 0.5)", width=0.5),
            marker=dict(
                color="rgba(255, 165, 0, 0.8)",
                size=24,
                line=dict(color="black", width=1.5),
            ),
            text=feature_scores_df["Feature"].apply(
                lambda x: hover_info[x.split("[")[0]] if "aggregate_" in x else x
            ),
            hoverinfo="text",
        )
    )

    for i in range(len(feature_scores_df)):
        fig.add_annotation(
            x=feature_scores_df["Importance"][i],
            y=i,
            text=f"{feature_scores_df['Importance'][i]:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
        )

    for feature in hover_info:
        top_features = hover_info[feature].split(", ")
        expected_feature_name = (
            truncate_label(f"{feature}[{top_contributors[feature]}]")
            + f" <span style='color:green;'>+{feature_count[feature]}</span>"
        )
        if expected_feature_name in feature_scores_df["Feature"].values:
            y_index = feature_scores_df[
                feature_scores_df["Feature"] == expected_feature_name
            ].index[0]
            for top_feature in top_features[:5]:
                if top_feature in feature_scores.index:
                    importance = feature_scores[top_feature]
                    fig.add_trace(
                        go.Scatter(
                            x=[importance],
                            y=[y_index],
                            mode="markers",
                            marker=dict(color="rgba(0, 128, 0, 0.6)", size=16),
                            showlegend=False,
                            hovertext=top_feature,
                            hoverinfo="text",
                        )
                    )

    for i, feature in enumerate(feature_scores_df["Feature"]):
        if "aggregate_" in feature:
            original_feature = feature.split("[")[0]
            if original_feature in feature_medians:
                fig.add_trace(
                    go.Scatter(
                        x=[
                            feature_medians[original_feature],
                            feature_medians[original_feature],
                        ],
                        y=[i - 0.5, i + 0.5],
                        mode="lines",
                        line=dict(color="red", width=1.5, dash="dot"),
                        name=f"{original_feature} Median",
                        hoverinfo="text",
                        hovertext=f"{original_feature} Median: {feature_medians[original_feature]:.2f}",
                    )
                )

    non_aggregate_median = feature_scores.loc[
        ~feature_scores.index.str.startswith("aggregate_")
    ].median()

    fig.add_trace(
        go.Scatter(
            x=[non_aggregate_median, non_aggregate_median],
            y=[-0.5, len(feature_scores_df) - 0.5],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name="Non-Aggregate Median",
            hoverinfo="text",
            hovertext=f"Non-Aggregate Median: {non_aggregate_median:.2f}",
            opacity=0.25,
        )
    )

    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance",
        yaxis=dict(
            tickvals=list(range(len(feature_scores_df))),
            ticktext=feature_scores_df["Feature"],
            showgrid=False,
            tickfont=dict(size=14, color="black"),
            titlefont=dict(size=16, color="black"),
        ),
        height=height,
        width=1600,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            tickfont=dict(size=14, color="black"),
            titlefont=dict(size=16, color="black"),
        ),
        font=dict(size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=300, r=50, t=70, b=40),
        shapes=zebra_shapes,
    )

    pio.write_html(fig, file=f"{output_filename}.html", auto_open=False)
    fig.write_image(f"{output_filename}.png")

    fig.show()


def plot_categorical_comparison(
    df,
    category_x,
    category_y,
    filter_non_frequent=False,
    threshold=0.05,
    relative_by_category=False,
    normalized=False,
    size_max=40,
    height=800,
    width=1200,
    title=None,
):
    """
    Plots a bubble chart comparing two categorical variables with options for relative percentages and normalization.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - category_x (str): The name of the categorical variable to be plotted on the x-axis.
    - category_y (str): The name of the categorical variable to be plotted on the y-axis.
    - filter_non_frequent (bool): If True, filters out categories with occurrences below the threshold.
    - threshold (float): The minimum proportion required to keep a category (default is 0.05, i.e., 5%).
    - relative_by_category (bool): If True, calculates relative percentages by category_x. If False, by category_y.
    - normalized (bool): If True, normalizes the data to account for category size imbalances.
    - size_max (int): Maximum size of the bubbles.
    - height (int): Height of the plot.
    - width (int): Width of the plot.
    - title (str): Title of the plot.
    # Example: Comparing 'cat1' and 'cat2' with relative percentage by 'cat2'
    plot_categorical_comparison(
        df,
        category_x='cat1',
        category_y='cat2',
        filter_non_frequent=True,
        threshold=0.05,
        relative_by_category=True,
        normalized=True,
        title='Normalized Relative Percentages of cat2 by cat1'
    )

    """

    import plotly.express as px
    import numpy as np

    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Ensure the categorical columns exist
    if category_x not in df.columns:
        raise ValueError(f"DataFrame does not contain '{category_x}' column")
    if category_y not in df.columns:
        raise ValueError(f"DataFrame does not contain '{category_y}' column")

    # Drop rows with NaN in the categorical columns
    df = df.dropna(subset=[category_x, category_y])

    # Optionally filter out infrequent categories based on the threshold
    if filter_non_frequent:
        # Filter category_x
        counts_x = df[category_x].value_counts(normalize=True)
        frequent_x = counts_x[counts_x >= threshold].index
        df = df[df[category_x].isin(frequent_x)]

        # Filter category_y
        counts_y = df[category_y].value_counts(normalize=True)
        frequent_y = counts_y[counts_y >= threshold].index
        df = df[df[category_y].isin(frequent_y)]

    if relative_by_category:
        # Calculate relative percentage by category_x (across the entire dataset)
        df["Total_in_X"] = df.groupby(category_x)[category_y].transform("count")
        df["Count_in_Y"] = df.groupby([category_x, category_y])[category_y].transform(
            "count"
        )

        if normalized:
            # Normalize based on the size of category_y to prevent large categories from dominating
            df["Total_in_Y"] = df.groupby(category_y)[category_x].transform("count")
            df["Weighted_Contribution"] = (df["Count_in_Y"] / df["Total_in_X"]) * (
                1 / df["Total_in_Y"]
            )
            df["Relative_Percentage"] = df["Weighted_Contribution"] * 100
        else:
            # Normal relative percentage calculation
            df["Relative_Percentage"] = (df["Count_in_Y"] / df["Total_in_X"]) * 100
    else:
        # Calculate relative percentage by category_y (within each category_y)
        df["Total_in_Y"] = df.groupby(category_y)[category_x].transform("count")
        df["Count_in_X"] = df.groupby([category_y, category_x])[category_x].transform(
            "count"
        )
        df["Relative_Percentage"] = (df["Count_in_X"] / df["Total_in_Y"]) * 100

    # Replace NaN values with 0 in Relative_Percentage
    df["Relative_Percentage"] = df["Relative_Percentage"].replace(np.nan, 0)

    # Create a bubble chart
    fig_bubble = px.scatter(
        df,
        x=category_x,
        y=category_y,
        size="Relative_Percentage",
        color=category_y,
        hover_data={
            "Relative_Percentage": ":.2f",
            category_x: True,
            category_y: True,
            "Count_in_Y" if relative_by_category else "Count_in_X": True,
            "Total_in_X" if relative_by_category else "Total_in_Y": True,
        },
        title=title or f"Relative Percentages of {category_x} by {category_y}",
        labels={
            category_x: category_x.replace("_", " ").title(),
            category_y: category_y.replace("_", " ").title(),
        },
        size_max=size_max,
    )

    # Customize layout for better visuals
    fig_bubble.update_layout(
        xaxis_title=category_x.replace("_", " ").title(),
        yaxis_title=category_y.replace("_", " ").title(),
        legend_title=category_y.replace("_", " ").title(),
        showlegend=True,
        height=height,
        width=width,
        margin=dict(l=40, r=40, t=80, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Rockwell"),
    )

    # Display the plot
    fig_bubble.show()

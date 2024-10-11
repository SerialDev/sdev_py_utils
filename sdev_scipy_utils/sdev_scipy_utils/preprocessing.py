#


def url_tokenizer(input):
    """
    * type-def ::(str) -> List[str]
    * ---------------{Function}---------------
        * Tokenizes a URL string by splitting it based on "/", ".", and "-" delimiters.
    * ----------------{Returns}---------------
        * : tokens ::List[str] | A list of unique tokens from the input URL string
    * ----------------{Params}----------------
        * : input ::str | The URL string to tokenize
    * ----------------{Usage}-----------------
        * >>> url_tokenizer("https://www.example.com/some-page.html")
        * ["https:", "www", "example", "some", "page", "html"]
    * ----------------{Notes}-----------------
        * This function can be useful for extracting meaningful tokens from a URL for further processing or analysis.
    """
    tokens_by_slash = str(input.encode("utf-8")).split("/")
    all_tokens = set()
    for token_slash in tokens_by_slash:
        tokens = token_slash.split("-")
        for token in tokens:
            tokens_by_dot = token.split(".")
            all_tokens.update(tokens_by_dot)
    all_tokens.discard("com")
    return list(all_tokens)
    # tokensBySlash = str(input.encode("utf-8")).split("/")
    # allTokens = []
    # for i in tokensBySlash:
    #     tokens = str(i).split("-")
    #     tokensByDot = []
    #     for j in range(0, len(tokens)):
    #         tempTokens = str(tokens[j]).split(".")
    #         tokentsByDot = tokensByDot + tempTokens
    #     allTokens = allTokens + tokens + tokensByDot
    # allTokens = list(set(allTokens))
    # if "com" in allTokens:
    #     allTokens.remove("com")
    # return allTokens


def split_keep(s, delimiter):
    """
    * type-def ::(str, str) -> List[str]
    * ---------------{Function}---------------
        * Splits a string by the delimiter, keeping the delimiter at the end of each substring.
    * ----------------{Returns}---------------
        * : split_list ::List[str] | A list of substrings with the delimiter appended
    * ----------------{Params}----------------
        * : s ::str | The string to be split
        * : delimiter ::str | The delimiter to split the string by
    * ----------------{Usage}-----------------
        * >>> split_keep("a,b,c,d", ",")
        * ["a,", "b,", "c,", "d"]
    * ----------------{Notes}-----------------
        * This function can be useful when the delimiter is needed for further processing.
    """
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]


def parse_uri(s):
    """
    * type-def ::(str) -> List[List[List[str]]]
    * ---------------{Function}---------------
        * Parses a URI string, creating a nested list structure based on "/", "?", and "&" delimiters.
    * ----------------{Returns}---------------
        * : parsed_uri ::List[List[List[str]]] | A nested list representing the parsed URI
    * ----------------{Params}----------------
        * : s ::str | The URI string to parse
    * ----------------{Usage}-----------------
        * >>> parse_uri("example.com/foo?bar=1&baz=2")
        * [[["example.com"], ["foo"], ["bar=1", "baz=2"]]]
    * ----------------{Notes}-----------------
        * This function can be used to parse and analyze a URI string more conveniently.
    """
    return [
        [[split_keep(z, "=") for z in split_keep(y, "&")] for y in split_keep(x, "?")]
        for x in s.split("/")
    ]


def cluster_single_column(
    data, n_clusters=3, cardinality_threshold=10, n_neighbors=30, min_cluster_size=10
):
    """
    * type-def ::(pd.Series, int, int, int, int) -> np.ndarray
    * ---------------{Function}---------------
        * Clusters the input data column (Series) using UMAP for dimensionality reduction and HDBSCAN for clustering.
    * ----------------{Returns}---------------
        * : labels ::np.ndarray | An array of cluster labels for each data point in the input data column.
    * ----------------{Params}----------------
        * : data ::pd.Series | The input data column (Series) to be clustered.
        * : n_clusters ::int | The number of clusters to generate. Default is 3.
        * : cardinality_threshold ::int | The threshold for using one-hot encoding for categorical data. Default is 10.
        * : n_neighbors ::int | The number of neighbors to consider when constructing the UMAP graph. Default is 30.
        * : min_cluster_size ::int | The minimum size of clusters allowed by HDBSCAN. Default is 10.
    * ----------------{Usage}-----------------
        * >>> data = pd.Series(...)
        * >>> cluster_labels = cluster_single_column(data, n_clusters=5, cardinality_threshold=8, n_neighbors=20, min_cluster_size=15)
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * umap-learn
          * hdbscan
          * scikit-learn
          * pandas
          * numpy
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input data column
          * and the chosen parameters. Large data columns or small n_neighbors values may increase the
          * computation time. Adjust the parameters accordingly to balance performance and clustering quality.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input data.
    * ----------------{Big O Complexity}------
        * Time Complexity:
          * UMAP has a time complexity of O(N log N) for N data points, where N is the number of data points in the input data column.
          * HDBSCAN has a time complexity of O(N log N) for N data points in the worst case.
          * Therefore, the overall time complexity of this function is approximately O(N log N), where N is the number of data points in the input data column.
        * Space Complexity:
          * The space complexity of this function is determined by the memory required to store the encoded, scaled, and transformed data, as well as the memory needed for the UMAP and HDBSCAN algorithms.
          * The space complexity of UMAP is O(N), where N is the number of data points in the input data column.
          * The space complexity of HDBSCAN is also O(N).
          * Therefore, the overall space complexity of this function is approximately O(N), where N is the number of data points in the input data column.
    """
    import umap
    import hdbscan
    from sklearn.preprocessing import MinMaxScaler

    if data.dtype == "object" or pd.api.types.is_categorical_dtype(data):
        unique_count = data.nunique()
        if unique_count <= cardinality_threshold:
            encoder = OneHotEncoder(sparse=False)
            encoded_data = encoder.fit_transform(data.to_numpy().reshape(-1, 1))
        else:
            encoder = LabelEncoder()
            encoded_data = encoder.fit_transform(data).astype(np.float64).reshape(-1, 1)
    else:
        encoded_data = data.to_numpy().reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(encoded_data)

    print(f"scaled_data.shape: {scaled_data.shape}")
    print(f"scaled_data.dtype: {scaled_data.dtype}")

    reducer = umap.UMAP(n_neighbors=n_neighbors)
    try:
        embedding = reducer.fit_transform(scaled_data)
    except Exception as e:
        print(f"Error: {e}")
        print(f"scaled_data: {scaled_data}")
        raise e

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    try:
        labels = clusterer.fit_predict(embedding)
    except Exception as e:
        print(f"Error: {e}")
        print(f"embedding: {embedding}")
        raise e

    return labels


def dtc_binning(X, y, class_names=["0", "1"], force_bins=0):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn import tree
    import matplotlib.pyplot as plt

    params = {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 3, 5, 10, 20]}
    clf_dt = DecisionTreeClassifier()
    clf = GridSearchCV(clf_dt, param_grid=params, scoring="accuracy")
    clf.fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 10))  # whatever size you want
    if force_bins > 0:
        clf_dt = DecisionTreeClassifier(max_depth=force_bins)
        clf_dt.fit(X, y)
    else:
        clf_dt = DecisionTreeClassifier(**clf.best_params_)
        clf_dt.fit(X, y)
    result = tree.plot_tree(
        clf_dt,
        filled=True,
        feature_names=list(X.columns),
        class_names=class_names,
        ax=ax,
    )
    return clf_dt, result, fig


def dtc_splits_binning(result_tree):
    split_list = [
        x
        for x in list(map(lambda x: x._text.split("\n")[0], result_tree))
        if "gini" not in x
    ]
    return split_list


def dtc_pd_binning(txt_result, df, col):
    import re

    txt_num = list(map(lambda x: int(x), set(re.findall(r"\d+", " ".join(txt_result)))))
    txt_num.sort()
    bins = txt_num

    labels = []
    for idx, i in enumerate(zip(bins, bins[1:])):
        if idx == 0:
            labels.append(f"{col} < {i[0]}")
        elif idx == len(bins) - 2:
            labels.append(f"{col} > {i[1]}")
        else:
            labels.append(f"{col} {i[0]}-{i[1]}")

    return bins, labels, pd.cut(df[col], bins=bins, labels=labels)


def dtc_full_binning(
    df, col_1, col_2, fill_val=0, class_names=["0", "1"], force_bins=0
):
    """
    * ---------------Typedef----------------
    * type-def ::(df: pandas.DataFrame, col_1: str, col_2: str, fill_val: int, class_names: list, force_bins: int) -> pandas.DataFrame

    * ---------------Function---------------
    * Performs decision tree classification binning on a pandas DataFrame
    * This function takes in a DataFrame and two column names, performs binning using decision trees,
    * and returns a new DataFrame with the binned values.

    * ---------------Returns---------------
    * -> pandas.DataFrame: A new DataFrame with the binned values.

    * ---------------Params----------------
    * df: pandas.DataFrame - The input DataFrame.
    * col_1: str - The first column name for binning.
    * col_2: str - The second column name for binning.
    * fill_val: int - The value to fill missing values with. Default is 0.
    * class_names: list - The names of the classes. Default is ["0", "1"].
    * force_bins: int - The number of bins to force. Default is 0.

    * ---------------Usage-----------------
    * df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [1, 1, 1, 2, 2]})
    * result = dtc_full_binning(df, 'A', 'B')

    * ---------------Notes-----------------
    * This function assumes that the input DataFrame has the two specified columns.
    * The function uses the dtc_binning and dtc_splits_binning functions internally.
    * The function returns a new DataFrame with the binned values.
    """
    subset = df[[col_1, col_2]]
    subset.fillna(0, inplace=True)
    X = subset[[col_1]]
    y = subset[col_2]
    clf, result_tree, tree_fig = dtc_binning(
        X, y, class_names=class_names, force_bins=force_bins
    )
    txt_result = dtc_splits_binning(result_tree)
    return dtc_pd_binning(txt_result, df, col_1)


def stratified_sample(data, target_column, sample_size):
    """
    * ---------------Typedef----------------
    * stratified_sample :: (pandas DataFrame, string, int) -> pandas DataFrame

    * ---------------Function---------------
    * Returns a stratified sample of the dataset with the specified sample size.
    * This ensures that the sample maintains the same class distribution as the original dataset.

    * ----------------Params----------------
    * data :: pandas DataFrame, pandas DataFrame containing your dataset.
    * target_column :: string, the name of your target column.
    * sample_size :: int, the desired number of samples in the output dataset.

    * ----------------Returns---------------
    * -> stratified_data :: pandas DataFrame, pandas DataFrame containing the stratified sample.

    * ----------------Usage-----------------
    * Call this function with a pandas DataFrame, a target column name, and a desired sample size to get a stratified sample of the data.
    * Example: stratified_data = stratified_sample(my_data, 'target_column_name', 1000)

    * ----------------Notes-----------------
    * This function ensures that the sample maintains the same class distribution as the original dataset.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if sample_size >= len(data):
        raise ValueError(
            "Sample size must be less than the total number of samples in the dataset."
        )

    # Prepare the splitter
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=42)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Perform stratified sampling
    for _, test_index in sss.split(X, y):
        stratified_data = data.iloc[test_index]
        return stratified_data

    # If no split was made, raise an error
    raise RuntimeError("Failed to generate stratified sample.")

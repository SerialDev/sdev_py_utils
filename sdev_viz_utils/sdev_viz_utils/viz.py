import seaborn as sns


def seaborn_setup():
    '''
* ---------------Function---------------
* Initializes seaborn settings for visualizations
* ----------------Returns---------------
* -> None
* ----------------Params----------------
* None
* ----------------Usage-----------------
* seaborn_setup()
    '''
    sns.set(style="whitegrid", palette="muted")

def cat_factorplot(data, category_col="category", value_col="value"):
    """
    * type-def ::(pd.DataFrame, str, str) -> seaborn.axisgrid.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn catplot based on the input DataFrame. The function creates a horizontal or vertical
          * catplot depending on the number of unique categories in the specified category column.
    * ----------------{Returns}---------------
        * : result ::seaborn.axisgrid.FacetGrid | The resulting catplot.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> catplot = cat_factorplot(df, category_col="custom_category", value_col="custom_value")
    * ----------------{Output}----------------
        * A catplot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Notes}-----------------
        * This function is useful for creating catplots with a dynamic orientation depending on the number of unique categories.
          * If there are more than 10 unique categories, the catplot will be horizontal; otherwise, it will be vertical.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """

    num_unique_categories = data[category_col].nunique()

    if num_unique_categories > 10:
        result = sns.catplot(
            x=value_col, y=category_col, palette=["r", "c", "y"], data=data, kind="bar"
        )
    else:
        result = sns.catplot(
            x=category_col, y=value_col, palette=["r", "c", "y"], data=data, kind="bar"
        )
    return result



def cat_swarmplot(data):
    """
* ---------------Function---------------
* Draw a categorical scatterplot to show each observation
* ----------------Returns---------------
* -> result ::AxesSubplot  
* ----------------Params----------------
* data :: pandas.DataFrame | Dataframe containing the data to plot
* ----------------Usage-----------------
* Use this function to create a categorical scatterplot.
* The function will automatically adjust the x and y axes based on the length of the "category" column in the data.
* The palette used is ["r", "c", "y"].
* You can customize the plot further by using Seaborn's swarmplot options.
    """
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.swarmplot(
            x="value", y="category", palette=["r", "c", "y"], data=data
        )
    else:
        result = sns.swarmplot(
            x="category", y="value", palette=["r", "c", "y"], hue="category", data=data
        )
    return result


def cat_manhattan(data, category_col="category", value_col="value", facet=None):
    """
    * type-def ::(pd.DataFrame, str, str) -> matplotlib.axes.Axes
    * ---------------{Function}---------------
        * Creates a Seaborn swarmplot with a log-scaled x-axis based on the input DataFrame. The function creates
          * a horizontal or vertical swarmplot depending on the number of unique categories in the specified
          * category column.
    * ----------------{Returns}---------------
        * : result ::matplotlib.axes.Axes | The resulting swarmplot.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> swarmplot = cat_manhattan(df, category_col="custom_category", value_col="custom_value")
    * ----------------{Output}----------------
        * A swarmplot visualization with horizontal or vertical orientation depending on the number of unique categories,
          * and a log-scaled x-axis.
    * ----------------{Notes}-----------------
        * This function is useful for creating swarmplots with a dynamic orientation depending on the number of unique
          * categories and a log-scaled x-axis. If there are more than 10 unique categories, the swarmplot will be
          * horizontal; otherwise, it will be vertical.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """

    num_unique_categories = data[category_col].nunique()

    if num_unique_categories > 10:
        result = sns.swarmplot(
            x=value_col, y=category_col, palette=["r", "c", "y"], data=data
        )
        result.set(xscale="log")
    else:
        result = sns.swarmplot(
            x=category_col, y=value_col, palette=["r", "c", "y"], hue=facet, data=data
        )
        result.set(xscale="log")
    return result


def cat_manhattan_bar(data, category_col="category", value_col="value", height=5, facet=None):
    """
    * type-def ::(pd.DataFrame, str, str, int) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn bar plot with a log-scaled x-axis based on the input DataFrame. The function creates
          * a horizontal or vertical bar plot depending on the number of unique categories in the specified
          * category column.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting bar plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : height ::int | The height of each facet in inches. Default is 5.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> barplot = cat_manhattan_bar(df, category_col="custom_category", value_col="custom_value")
    * ----------------{Output}----------------
        * A bar plot visualization with horizontal or vertical orientation depending on the number of unique categories,
          * and a log-scaled x-axis.
    * ----------------{Notes}-----------------
        * This function is useful for creating bar plots with a dynamic orientation depending on the number of unique
          * categories and a log-scaled x-axis. If there are more than 10 unique categories, the bar plot will be
          * horizontal; otherwise, it will be vertical.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """

    num_unique_categories = data[category_col].nunique()

    if num_unique_categories > 10:
        result = sns.catplot(
            x=value_col,
            y=category_col,
            kind="bar",
            palette=["r", "c", "y"],
            data=data,
            height=height,
            aspect=2,  # height should be two times width
        )
        result.set(xscale="log")
    else:
        result = sns.catplot(
            x=category_col,
            y=value_col,
            kind="bar",
            palette=["r", "c", "y"],
            hue=facet,
            data=data,
            height=height,
            aspect=2,  # height should be two times width
        )
        result.set(xscale="log")
    return result

def cat_boxen(data, category_col="category", value_col="value", facet=None):
    """
    * type-def ::(pd.DataFrame, str, str) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn boxen plot based on the input DataFrame. The function creates
          * a horizontal or vertical boxen plot depending on the number of unique categories in the specified
          * category column.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting boxen plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> boxen_plot = cat_boxen(df, category_col="custom_category", value_col="custom_value")
    * ----------------{Output}----------------
        * A boxen plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Notes}-----------------
        * This function is useful for creating boxen plots with a dynamic orientation depending on the number of unique
          * categories. If there are more than 10 unique categories, the boxen plot will be
          * horizontal; otherwise, it will be vertical.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """

    num_unique_categories = data[category_col].nunique()

    if num_unique_categories > 10:
        result = sns.catplot(
            x=value_col,
            y=category_col,
            kind="boxen",
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        result = sns.catplot(
            x=category_col,
            y=value_col,
            kind="boxen",
            palette=["r", "c", "y"],
            hue=facet,
            data=data,
        )
    return result

def cat_box(data):
    '''
    * ---------------Function---------------
* Draws a categorical scatterplot to show each observation
* ----------------Returns---------------
* -> result ::matplotlib AxesSubplot
* ----------------Params----------------
* data ::dict | dictionary containing the data to be plotted
* ----------------Usage-----------------
* cat_box(data)
    '''
    # Draw a categorical scatterplot to show each observation
    if len(data["category"]) > 10:
        result = sns.catplot(
            x="value", y="category", kind="box", palette=["r", "c", "y"], data=data
        )
    else:
        result = sns.catplot(
            x="category",
            y="value",
            kind="box",
            palette=["r", "c", "y"],
            hue="category",
            data=data,
        )
    return result


def cat_strip(
    data,
    category_col="category",
    value_col="value",
    facet=None,
    jitter=False,
):
    """
    * type-def ::(pd.DataFrame, str, str, Union[None, str], bool) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn strip plot based on the input DataFrame. The function creates
          a horizontal or vertical strip plot depending on the number of unique categories in the specified
          category column. The jitter parameter controls whether the data points are jittered or not.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting strip plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : facet ::Union[None, str] | The name of the column to use for facetting. Default is None.
        * : jitter ::bool | Whether to apply jitter to the data points. Default is False.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> strip_plot = cat_strip(df, category_col="custom_category", value_col="custom_value", jitter=True)
    * ----------------{Output}----------------
        * A strip plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Notes}-----------------
        * This function is useful for creating strip plots with a dynamic orientation depending on the number of unique
          categories. If there are more than 10 unique categories, the strip plot will be
          horizontal; otherwise, it will be vertical.
    * ----------------{Dependencies}---------
        * seaborn
        * pandas
    * ----------------{Performance}-----------
        * Primarily depends on input DataFrame size and machine rendering capabilities.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * Does not modify the input DataFrame.
    """

    # Draw a categorical scatterplot to show each observation
    if len(data[category_col].unique()) > 10:
        result = sns.catplot(
            x=value_col,
            y=category_col,
            col=facet,
            jitter=jitter,
            data=data,
        )
    else:
        result = sns.catplot(
            x=category_col,
            y=value_col,
            jitter=jitter,
            data=data,
        )
    return result

def cat_violin(
    data,
    category_col="category",
    value_col="value",
    split=False,
    facet=None,
    inner="stick",
    palette=["r", "c", "y"],
):
    """
    * type-def ::(pd.DataFrame, str, str, bool, Union[None, str], str, List[str]) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn violin plot based on the input DataFrame. The function creates
          a horizontal or vertical violin plot depending on the number of unique categories in the specified
          category column. The split and inner parameters provide additional customization options.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting violin plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : split ::bool | Whether to split the violins for easier comparison. Default is False.
        * : facet ::Union[None, str] | The name of the column to use for facetting. Default is None.
        * : inner ::str | The representation of the datapoints in the violin interior. Default is 'stick'.
        * : palette ::List[str] | The color palette to use for the plot. Default is ["r", "c", "y"].
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> violin_plot = cat_violin(df, category_col="custom_category", value_col="custom_value", split=True, inner="box")
    * ----------------{Output}----------------
        * A violin plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Notes}-----------------
        * This function is useful for creating violin plots with a dynamic orientation depending on the number of unique
          categories. If there are more than 10 unique categories, the violin plot will be
          horizontal; otherwise, it will be vertical.
    * ----------------{Dependencies}---------
        * seaborn
        * pandas
    * ----------------{Performance}-----------
        * Primarily depends on input DataFrame size and machine rendering capabilities.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * Does not modify the input DataFrame.
    """

    if len(data[category_col].unique()) > 10:
        result = sns.catplot(
            x=value_col,
            y=category_col,
            kind="violin",
            col=facet,
            palette=palette,
            inner=inner,
            split=split,
            data=data,
        )
    else:
        result = sns.catplot(
            x=category_col,
            y=value_col,
            kind="violin",
            col=facet,
            palette=palette,
            inner=inner,
            hue=category_col,
            split=split,
            data=data,
        )
    return result

def cat_dist_swarm(data, category_col="category", value_col="value", inner="violin", facet=None):
    """
    * type-def ::(pd.DataFrame, str, str, str, Union[None, str]) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn categorical distribution plot combined with a swarm plot.
          * The function supports "violin" and "box" plots and adjusts the orientation depending on the number of unique categories
          * in the specified category column.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting combined distribution and swarm plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : inner ::str | The type of distribution plot to use, either 'violin' or 'box'. Default is 'violin'.
        * : facet ::Union[None, str] | The name of the column to use for facetting. Default is None.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> combined_plot = cat_dist_swarm(df, category_col="custom_category", value_col="custom_value", inner="box")
    * ----------------{Output}----------------
        * A combined distribution and swarm plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * seaborn
          * pandas
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input DataFrame and the rendering
          * capabilities of the machine running the code. For large DataFrames or machines with limited resources, consider
          * using more efficient visualization libraries or reducing the size of the input data.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """
    x_val = value_col if len(data[category_col]) > 10 else category_col
    y_val = category_col if len(data[category_col]) > 10 else value_col

    if inner not in ["violin", "box"]:
        raise ValueError("Invalid inner value. Supported values are 'violin' and 'box'.")

    result = sns.catplot(x=x_val, y=y_val, col=facet, kind=inner, inner=None, data=data)
    sns.swarmplot(x=x_val, y=y_val, color="k", size=3, data=data, ax=result.ax)

    return result



def cat_bar(data, category_col="category", value_col="value", facet=None):
    """
    * type-def ::(pd.DataFrame, str, str, Union[None, str]) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn categorical bar plot based on the input DataFrame. The function creates
          * a horizontal or vertical bar plot depending on the number of unique categories in the specified
          * category column.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting bar plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : facet ::Union[None, str] | The name of the column to use for facetting. Default is None.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> bar_plot = cat_bar(df, category_col="custom_category", value_col="custom_value")
    * ----------------{Output}----------------
        * A bar plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * seaborn
          * pandas
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input DataFrame and the rendering
          * capabilities of the machine running the code. For large DataFrames or machines with limited resources, consider
          * using more efficient visualization libraries or reducing the size of the input data.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """
    plt.clf()
    if len(data[category_col]) > 10:
        result = sns.catplot(
            x=value_col,
            y=category_col,
            kind="bar",
            col=facet,
            palette=["r", "c", "y"],
            data=data,
        )
    else:
        result = sns.catplot(
            x=category_col,
            y=value_col,
            kind="bar",
            palette=["r", "c", "y"],
            hue=category_col,
            data=data,
        )
    return result



def cat_count(data, category_col="category", value_col="value", reverse=False, facet=None):
    """
    * type-def ::(pd.DataFrame, str, str, bool, Union[None, str]) -> sns.FacetGrid
    * ---------------{Function}---------------
        * Creates a Seaborn categorical count plot based on the input DataFrame. The function creates
          * a horizontal or vertical count plot depending on the number of unique categories in the specified
          * category column and the 'reverse' parameter.
    * ----------------{Returns}---------------
        * : result ::sns.FacetGrid | The resulting count plot FacetGrid.
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : category_col ::str | The name of the column containing the categories. Default is 'category'.
        * : value_col ::str | The name of the column containing the values. Default is 'value'.
        * : reverse ::bool | Whether to reverse the orientation of the plot. Default is False.
        * : facet ::Union[None, str] | The name of the column to use for facetting. Default is None.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> count_plot = cat_count(df, category_col="custom_category", value_col="custom_value", reverse=True)
    * ----------------{Output}----------------
        * A count plot visualization with horizontal or vertical orientation depending on the number of unique categories.
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * seaborn
          * pandas
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input DataFrame and the rendering
          * capabilities of the machine running the code. For large DataFrames or machines with limited resources, consider
          * using more efficient visualization libraries or reducing the size of the input data.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """
    x_col = category_col if reverse else value_col
    hue_col = category_col if len(data[category_col]) <= 10 else None

    result = sns.catplot(
        x=x_col,
        col=facet,
        kind="count",
        palette=["r", "c", "y"],
        hue=hue_col,
        data=data,
    )
    
    return result


def b64_div(b64_img):
    '''
    * ---------------b64_div---------------
    * Returns a base64 encoded image tag
    * ----------------Returns---------------
    * -> str: A base64 encoded image tag
    * ----------------Params----------------
    * b64_img: str: The base64 encoded image
    * ----------------Usage-----------------
    * Use this function to convert a base64 encoded image to an HTML image tag
    '''
    return f'<img src="data:image/png;base64,{b64_img}">'


def convert_plt_b64(viz):
    '''
* ---------------convert_plt_b64---------------
* Converts a matplotlib visualization to a base64 encoded string
* ----------------Returns---------------
* -> str: A base64 encoded string representation of the visualization
* ----------------Params----------------
* viz: <any>: A matplotlib visualization
* ----------------Usage-----------------
* Use this function to convert a matplotlib visualization to a base64 encoded string
    '''
    import io
    import base64

    buffer = io.BytesIO()
    viz.savefig(buffer)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()


def sep_div(div):
    '''
* ---------------sep_div---------------
* Separates a list of div elements into a single div element
* ----------------Returns---------------
* -> str: A single div element containing all the input div elements
* ----------------Params----------------
* div: list: A list of div elements
* ----------------Usage-----------------
* Use this function to combine multiple div elements into a single div element
    '''
    def create_div(data):
        """
        * ---------------Function---------------
* Creates an HTML div element with the provided data
* ----------------Returns---------------
* -> str: an HTML div element as a string
* ----------------Params----------------
* data: <any> - the data to be wrapped in the div element
* ----------------Usage-----------------
* >>> create_div("Hello, World!")
* "<div> Hello, World! </div>"
        """
        return f"<div> {data} </div>"

    
    result = f""
    for i in div:
        result += create_div(i)
    return create_div(result)


def seaborn_multi_facet(
    df, core_facet, plot_type=cat_manhattan_bar, secondary_facet=None
):
    """
    * type-def ::(pd.DataFrame, str, Callable, Union[None, str]) -> str
    * ---------------{Function}---------------
        * Creates a multi-faceted plot using a provided plot type function, which should be compatible with Seaborn.
          * The plot is faceted based on the unique values of the 'core_facet' column.
    * ----------------{Returns}---------------
        * : result ::str | The resulting HTML div elements containing the base64 encoded images of the facets.
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The dataframe containing the data to be plotted.
        * : core_facet ::str | The name of the column containing the core facet categories.
        * : plot_type ::Callable | A plotting function compatible with Seaborn, such as cat_manhattan_bar.
        * : secondary_facet ::Union[None, str] | The name of the column containing the secondary facet categories. Default is None.
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame(...)
        * >>> facet_html = seaborn_multi_facet(df, core_facet="custom_core_facet", plot_type=custom_plot_type)
    * ----------------{Output}----------------
        * A string containing HTML div elements with base64 encoded images of the facets.
    * ----------------{Dependencies}---------
        * This function requires the following libraries:
          * seaborn
          * pandas
          * base64
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the size of the input DataFrame, the rendering
          * capabilities of the machine running the code, and the complexity of the plot type function. For large
          * DataFrames or machines with limited resources, consider using more efficient visualization libraries or
          * reducing the size of the input data.
    * ----------------{Side Effects}---------
        * None
    * ----------------{Mutability}------------
        * This function does not modify the input DataFrame.
    """
    result = []
    for i in list(df[core_facet].unique()):
        result.append(
            b64_div(
                convert_plt_b64(
                    plot_type(df[df[core_facet] == i][:100], facet=core_facet, height=5)
                )
            )
        )
    return sep_div(result)


def preprocess_viz_node(node, user):
    '''
* ---------------Function---------------
* Creates a pandas DataFrame from the input data.
* ----------------Returns---------------
* -> pandas.DataFrame : The created DataFrame with index set to a range of integers based on the length of the "category" column.
* ----------------Params----------------
* hdf : <any> : The input data to be converted into a DataFrame.
* ----------------Usage-----------------
* Call this function with a dictionary or similar data structure to create a pandas DataFrame with a numerical index.
* Example: h_df = create_dataframe(hdf)
    '''
    import distogram
    import pandas as pd

    hdf = {}
    hdf["category"] = []
    hdf["value"] = []
    hdf["user"] = []

    # TODO: add facet to sepparate based on zones/resources etc
    for i in range(len(node["nodes"])):
        hdf["category"].append(node["nodes"][i])
        hdf["value"].append(distogram.count(node[node["nodes"][i]]))
        hdf["user"].append(user)

    h_df = pd.DataFrame(hdf, index=list(range(len(hdf["category"]))))

    return h_df



import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def create_faceted_treemap(df, col, bin_size=10, facet_col=None, main_square_half_size=True, sunburst=False, constant_name=None):
    """
    Create faceted treemaps for a given DataFrame, column, and optional faceting column.
    The main treemap square can be made half the size of others.
    
    :param df: DataFrame containing the data.
    :param col: The name of the column to create a treemap for.
    :param bin_size: The size of bins for numerical data. Default is 10.
    :param facet_col: Optional column for faceting the treemaps. Default is None.
    :param main_square_half_size: If True, the main treemap square will be half the size. Default is True.
    :return: Plotly figure object.
    """
    # Check if the column is categorical and expand its categories if needed
    if pd.api.types.is_categorical_dtype(df[col]):
        if 'NaN found' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('NaN found')
    
    # Fill NaN values with a clear identifier
    df[col] = df[col].fillna('NaN found')
    if pd.api.types.is_numeric_dtype(df[col]):
        # Create bins
        df['bins'] = pd.cut(df[col], range(0, int(df[col].max() + bin_size), bin_size))
        col = 'bins'  # Update column to bin column

    if facet_col:
        # Determine the number of unique values in the facet column
        unique_values = df[facet_col].unique()
        n_facets = len(unique_values)
        # Adjust column widths
        col_widths = [0.5] + [1 for _ in range(n_facets - 1)] if main_square_half_size else [1] * n_facets

        # Create a subplot for each unique value in the facet column
        fig = make_subplots(rows=1, cols=n_facets, 
                            subplot_titles=[str(val) for val in unique_values],
                            specs=[[{'type': 'domain'} for _ in range(n_facets)]],
                            column_widths=col_widths)

        for i, value in enumerate(unique_values):
            temp_df = df[df[facet_col] == value]
            counts = temp_df[col].value_counts()
            if sunburst:
                treemap = go.Sunburst(labels=counts.index, parents=[""] * len(counts), values=counts.values)
            else:
                treemap = go.Treemap(labels=counts.index, parents=[""] * len(counts), values=counts.values)
            fig.add_trace(treemap, row=1, col=i + 1)

        fig.update_layout(title=f"Faceted Treemap of {col}")
        return fig
    else:
        # For non-faceted treemap
        counts = df[col].value_counts()
        if sunburst:
            fig = px.sunburst(counts, path=[px.Constant(f"{constant_name}"), counts.index], values=counts.name, title=f"Treemap of {col}")
        else:
            fig = px.treemap(counts, path=[px.Constant(f"{constant_name}"), counts.index], values=counts.name, title=f"Treemap of {col}")
        return fig


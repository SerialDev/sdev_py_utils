# Main Libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sci
import sklearn as sk
import shap


def table_header(cols):
    """
    * ---------------table_header---------------
* Generates HTML code for a table header
* ----------------Returns---------------
* -> result ::str
* ----------------Params----------------
* cols :: list[str] | A list of column headers
* ----------------Usage-----------------
* table_header(["Column 1", "Column 2", "Column 3"]) -> "<thead>...</thead>"
    """
    result = f"""
<thead>
    {table_row(cols, True)}
</thead> """
    return result


def table_footer(cols):
    """
    * ---------------table_footer---------------
* Generates HTML code for a table footer
* ----------------Returns---------------
* -> result ::str
* ----------------Params----------------
* cols :: list[str] | A list of column headers
* ----------------Usage-----------------
* table_footer(["Column 1", "Column 2", "Column 3"]) -> "<tfoot>...</tfoot>"
    """
    result = f"""
<tfoot>
    {table_row(cols, True)}
</tfoot> """
    return result


def table_row(dataset, header=False):
    """
    * ---------------table_row---------------
* Generates HTML code for a table row
* ----------------Returns---------------
* -> result ::str
* ----------------Params----------------
* dataset :: list[str] | A list of data
* header :: bool | Whether the row is a header row (default: False)
* ----------------Usage-----------------
* table_row(["Cell 1", "Cell 2", "Cell 3"], header=True) -> "<tr>...</tr>"
* table_row(["Cell 1", "Cell 2", "Cell 3"]) -> "<tr>...</tr>"
    """
    if header:
        tag = "th"
    else:
        tag = "td"
    data_row = ""
    for data in dataset:
        data_row += f" <{tag}>{data}</{tag}> "
    result = f"""
<tr>
    {data_row}
</tr> """
    return result


def table_body(data):
    """
    * ---------------table_body---------------
* Generates HTML code for a table body
* ----------------Returns---------------
* -> result ::str
* ----------------Params----------------
* data :: list[list[str]] | A 2D list of data
* ----------------Usage-----------------
* table_body([["Cell 1", "Cell 2", "Cell 3"], ["Cell 4", "Cell 5", "Cell 6"]]) -> "<tbody>...</tbody>"
    """
    result = " <tbody> "
    for i in data:
        result += table_row(i)
    result += " </tbody> "
    return result


def pd_html_table(df, fillna=True):
    """
    * ---------------Function---------------
* Generates an HTML table from a pandas DataFrame
* ----------------Returns---------------
* -> str: The generated HTML table
* ----------------Params---------------
* df :: DataFrame: The input DataFrame
* fillna :: bool: Whether to fill NaN values with '-' (default: True)
* ----------------Usage---------------
* >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
* >>> html_table = pd_html_table(df)
* print(html_table)
    """
    if fillna:
        df = df.fillna("-")
    cols = list(df.columns)
    contents = df.to_numpy()
    return f"""
<table class="table is-striped" style="padding-top: 20px;
    overflow: auto; height: 300px; max-height: 98%;">
    {table_header(cols)}

    {table_body(contents)}

    {table_footer(cols)}
</table>
"""


def b64_div(data):
    """
    * ---------------Function---------------
* Encodes a data URI for an image
* ----------------Returns---------------
* -> str: The encoded data URI
* ----------------Params---------------
* data :: <any>: The image data to encode
* ----------------Usage---------------
* >>> data = ...  # some image data
* >>> encoded_data = b64_div(data)
* print(encoded_data)
    """
    return f'<img src="data:image/png;base64,{data}" />'


def b64encode_buffer(buffer):
    """
    * ---------------Function---------------
* Encodes a buffer to a base64-encoded string
* ----------------Returns---------------
* -> str: The base64-encoded string
* ----------------Params---------------
* buffer :: Buffer: The buffer to encode
* ----------------Usage---------------
* >>> buffer = ...  # some buffer object
* >>> encoded_buffer = b64encode_buffer(buffer)
* print(encoded_buffer)
    """
    import base64

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def shap_summary(shap_values, explain_array, feature_names, max_display=20):
    """
* ---------------shap_summary---------------
* Generates a SHAP values summary plot
* ----------------Returns---------------
* -> fig ::matplotlib figure
* ----------------Params----------------
* shap_values ::<any>
* explain_array ::<any>
* feature_names ::list
* max_display ::int (default=20)
* ----------------Usage----------------
* Import necessary libraries and load data. 
* Call shap_summary function with shap values, explain array, feature names, and max display as arguments.
* A matplotlib figure object will be returned, which can be used for further customization or display.
    """
    plt.clf()
    plt.tight_layout()
    fig = shap.summary_plot(
        shap_values,
        top_anomalies_np,
        feature_names=df.columns.tolist(),
        max_display=max_display,
        sort=True,
        show=False,
    )
    plt.tight_layout()

    return fig


def bulma_inside_card(title, content, height_px=450, id_name=""):
    """
    * ---------------bulma_inside_card---------------
* Returns a formatted Bulma card HTML snippet
* ----------------Returns---------------
* -> str
* ----------------Params----------------
* title ::str
* content ::str
* height_px ::int (default=450)
* id_name ::str (default='')
* ----------------Usage-----------------
    """
    return f"""
<div class="card events-card">
  <header class="card-header">
    <p class="card-header-title">
    {title}
    </p><a href="#" class="card-header-icon" aria-label="more options">
     <span class="icon">
       <i class="fa fa-angle-down" aria-hidden="true"></i>
     </span>
    </a>
  </header>
  <div id="{id_name}"
    style="overflow: auto; height: {height_px}px; max-height: 98%;">
    {content}
  </div>
</div>
"""


def bulma_header():
    """
    **---------------Function---------------**
* Generate a Bulma CSS header
* 
* ----------------Returns---------------
* -> str | The HTML header code as a string
* ----------------Params----------------
* None
* ----------------Usage-----------------
* Call the function to generate a Bulma CSS header

def bulma_header():
    """
    return """

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>D&R dashboard</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,400,700" rel="stylesheet">
    <!-- Bulma Version 0.9.0-->
    <link rel="stylesheet" href="https://unpkg.com/bulma@0.9.0/css/bulma.min.css" />
    <link rel="stylesheet" type="text/css" href="../css/admin.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@2.8.0"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@0.5.7/chartjs-plugin-annotation.min.js"></script>
</head>

    """


def bulma_inside(content):
    """
    **---------------Function---------------**
* Generate a Bulma CSS HTML page with a given content
* 
* ----------------Returns---------------
* -> str | The full HTML page code as a string
* ----------------Params----------------
* content :: str | The content to be included in the HTML page
* ----------------Usage-----------------
* Call the function with the desired content to generate a full HTML page
    """
    return f"""
               <!DOCTYPE html>
           <html>
           {bulma_header()}
            <body>

{content}

            </body>

            """


print("Success")

# Main Libs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sci
import sklearn as sk
import shap


def table_header(cols):
    result = f"""
<thead>
    {table_row(cols, True)}
</thead> """
    return result


def table_footer(cols):
    result = f"""
<tfoot>
    {table_row(cols, True)}
</tfoot> """
    return result


def table_row(dataset, header=False):
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
    result = " <tbody> "
    for i in data:
        result += table_row(i)
    result += " </tbody> "
    return result


def pd_html_table(df, fillna=True):
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
    return f'<img src="data:image/png;base64,{data}" />'


def b64encode_buffer(buffer):
    import base64

    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def shap_summary(shap_values, explain_array, feature_names, max_display=20):
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
    return f"""
               <!DOCTYPE html>
           <html>
           {bulma_header()}
            <body>

{content}

            </body>

            """


print("Success")

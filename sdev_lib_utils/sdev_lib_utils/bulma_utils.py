#


def gen_query_string(var_name, content):
    return f"?{var_name}={content}"


def bulma_inside_col(data, val=6):
    return f"""
<div class="column is-{val}">
    {data}
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


def bulma_navbar_item(title, url):
    return f"""
<a class="navbar-item" href="{url}">
  {title}
</a>
    """


def bulma_navbar(
    title="Bulma Admin",
    title_url="../index.html",
    navbar_items=[("name", "admin.html")],
):
    navbar = ""
    for i, j in navbar_items:
        navbar += bulma_navbar_item(i, j)

    return f"""
<!-- START NAV -->
<nav class="navbar is-black">
  <div class="container">
    <div class="navbar-brand">
      <a class="navbar-item brand-text" href="{title_url}">
        {title}
      </a>
      <div class="navbar-burger burger" data-target="navMenu">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
    <div id="navMenu" class="navbar-menu">
      <div class="navbar-start">
        {navbar}
      </div>
    </div>
  </div>
</nav>
<!-- END NAV -->
    """


def bulma_list_items(list_items, active_item=None, ul=True):
    items = ""
    if active_item:
        items += f'<li><a class="is-active">{active_item}</a></li>'
    for i in list_items:
        items += f"<li><a>{i}</a></li>"

    return f"""
<ul class="menu-list">
   {items}
</ul>
    """


def bulma_side_container():
    return """
<div class="container">
  <div class="columns">
    <div class="column is-3">
      <aside class="menu is-hidden-mobile">
        <p class="menu-label">General</p>
        <ul class="menu-list">
          {bulma_list_items([customers, pickles], dashboard)}
        </ul>
        <p class="menu-label">Administration</p>
        <ul class="menu-list">
          <li>
            <a>Team Settings</a>
          </li>
          <li>
            <a>Manage Your Team</a>
            <ul>
              {bulma_list_items([Members, Plugins])}
            </ul>
          </li>
          <li>
            <a>Invitations</a>
          </li>
          <li>
            <a>Cloud Storage Environment Settings</a>
          </li>
          <li>
            <a>Authentication</a>
          </li>
          <li>
            <a>Payments</a>
          </li>
        </ul>
        <p class="menu-label">Transactions</p>
        <ul class="menu-list">
          <li>
            <a>Payments</a>
          </li>
          <li>
            <a>Transfers</a>
          </li>
          <li>
            <a>Balance</a>
          </li>
          <li>
            <a>Reports</a>
          </li>
        </ul>
      </aside>
    </div>
  </div>
</div>
"""


def bulma_breadcrumbs(current_breadcrumb, breadcrumb_list):
    bread = ""
    for i in breadcrumb_list:
        bread += f'<li><a href="../">{i}</a></li>'
    bread += f'<li class="is-active"><a href="#" aria-current="page">{current_breadcrumb}</a></li>'

    return f"""
<nav class="breadcrumb" aria-label="breadcrumbs">
  <ul>
    {bread}
  </ul>
</nav>
"""


def bulma_href(data, content=""):
    return f"""
<a href="{data}">{content}</a>
"""


def bulma_hero(title, subtitle):
    return f"""
<section class="hero is-info welcome is-small">
  <div class="hero-body">
    <div class="container">
      <h1 class="title">{title}</h1>
      <h2 class="subtitle">{subtitle}</h2>
    </div>
  </div>
</section>
    """


def bulma_info_tiles(tile_list=[("test", "this")]):
    tiles = ""
    for i, j in tile_list:
        tiles += f"""
<div class="tile is-parent">
  <article class="tile is-child box">
    <p class="title">{i}</p>
    <p class="subtitle">{j}</p>
  </article>
</div>
"""

    return f"""
<section class="info-tiles">
  <div class="tile is-ancestor has-text-centered">
    {tiles}
  </div>
</section>
"""


def bulma_tbody(row_contents=["test", "this"], cf_auth="", np_token=""):
    table = ""
    for i in row_contents:
        table += f'<td width="5%"> {i} </td>'

    return f"""
<tbody>
  <tr>
    <td width="5%"><i class="fa fa-bell-o"></i></td>
      {table}
    <td class="level-right"><a class="button is-small is-primary" href={row_contents[0] + "/" + gen_query_string("cf_auth",cf_auth) + "&" +gen_query_string("np_token", np_token)[1:]}>Action</a></td>
  </tr>
</tbody>
    """


def bulma_search_card(search_header):
    return f"""
<div class="card">
  <header class="card-header">
    <p class="card-header-title">{search_header}</p><a href="#" class="card-header-icon" aria-label="more options">
    <span class="icon">
    <i class="fa fa-angle-down" aria-hidden="true"></i>
    </span>
    </a>
  </header>
  <div class="card-content">
    <div class="content">
      <div class="control has-icons-left has-icons-right">
        <input class="input is-large" type="text" placeholder="">
    <span class="icon is-medium is-left">
    <i class="fa fa-search"></i>
    </span>
    <span class="icon is-medium is-right">
    <i class="fa fa-check"></i>
    </span>
      </div>
    </div>
  </div>
</div>
"""


def bulma_events_card(title, rows_list=[bulma_tbody(["why", "does", "this", "work"])]):
    rows = ""
    for i in rows_list:
        rows += i

    return f"""
<div class="card events-card">
  <header class="card-header">
    <p class="card-header-title">{title}</p><a href="#" class="card-header-icon" aria-label="more options">
    <span class="icon">
    <i class="fa fa-angle-down" aria-hidden="true"></i>
    </span>
    </a>
  </header>
  <div class="card-table">
    <div class="content">
      {rows}
      <table class="table is-fullwidth is-striped"></table>
    </div>
  </div>
  <footer class="card-footer">
    <a href="#" class="card-footer-item">Export All</a>
  </footer>
</div>
"""


def bulma_inside_card(title, content, height_px=300, id_name=""):
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


def maybe_line_added(data_length, line_added=False):
    if line_added is False:
        result = []
        for i in range(data_length):
            result.append("horizontalBar")
    else:
        result = []
        for i in range(data_length):
            result.append("bar")
        result[-1] = "line"
    return result


def chartjs_data(
    data_titles,
    labels,
    datasets,
    chart_type=None,
    rgbs=None,
    min_bar_length=1,
    max_bar_thickness=8,
):
    import json

    if not rgbs:
        rgbs = []
        for idx, i in enumerate(data_titles):
            rgbs.append(
                (
                    np.random.randint(0, 1),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                )
            )

    if chart_type is None:
        result = []
        for idx, i in enumerate(data_titles):
            result.append(
                {
                    "label": data_titles[idx],
                    "backgroundColor": f"rgb{rgbs[idx]}",
                    "borderColor": "rgb(255,99,132)",
                    "data": datasets,
                    "minBarLenght": min_bar_length,
                    "maxBarThickness": max_bar_thickness,
                }
            )

    else:
        result = []
        for idx, i in enumerate(data_titles):
            result.append(
                {
                    "label": data_titles[idx],
                    "backgroundColor": f"rgb{rgbs[idx]}",
                    "borderColor": f"rgb({np.random.randint(0, 1)},{np.random.randint(0, 255)},{np.random.randint(0, 255)})",
                    "data": datasets[idx],
                    "type": chart_type[idx],
                    "fill": False,
                    "minBarLenght": min_bar_length,
                    "maxBarThickness": max_bar_thickness,
                }
            )

    return json.dumps(
        {
            "labels": labels,
            "datasets": result,
        }
    )


def chartjs_bar(dataset, height=300, chart_type="horizontalBar", p_value=0):
    chart_id = np.random.randint(0, 100)

    if p_value > 0:
        result = f"""
    <canvas id="{chart_id}" height="{height}"></canvas>
    <script>

    var ctx = document.getElementById('{chart_id}').getContext('2d');
var chart = new Chart(ctx, {{
    // The type of chart we want to create
    type: '{chart_type}',

    // The data for our dataset
    data: {dataset} ,

    // Configuration options go here
    options: {{
        responsive: true,
		annotation: {{
			drawTime: 'afterDatasetsDraw',
			events: ['click'],
			dblClickSpeed: 350,
			annotations: [{{
				drawTime: 'afterDraw',
				id: 'a-line-1',
				type: 'line',
				mode: 'horizontal',
				scaleID: 'y-axis-0',
				value: {p_value},
				borderColor: 'red',
				borderWidth: 2,
				onClick: function(e) {{
				}}
			}}]
		}}
    }}
}});

    </script>

    """
    else:
        result = f"""
    <canvas id="{chart_id}" height="{height}"></canvas>
    <script>

    var ctx = document.getElementById('{chart_id}').getContext('2d');
var chart = new Chart(ctx, {{
    // The type of chart we want to create
    type: '{chart_type}',

    // The data for our dataset
    data: {dataset} ,

    // Configuration options go here
    options: {{

    }}
}});

    </script>

    """

    return result


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


def list_html_table(cols, rows):
    return f"""
<table class="table is-striped">
    {table_header(cols)}

    {table_body(rows)}

    {table_footer(cols)}

</table>
"""


def pd_html_table(df):
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


def log10_(data):
    try:
        result = log10(data)
    except Exception:
        result = 0
    return result


def paginated_html(contents):
    import uuid

    def id_generator(
        size=6,
    ):
        import string
        import random

        chars = string.ascii_lowercase
        return "".join(random.choice(chars) for _ in range(size))

    number_pages = [uuid.uuid4().hex for i in range(len(contents))]
    pages_uuid = id_generator(12)

    pagination = ' <ul class="pagination-list"> '
    hidden_divs = ""
    for idx, i in enumerate(number_pages):
        if idx == 0:
            pagination += f' <button class="button is-small " href="  # " onclick=showPages{pages_uuid}("{i}")> {idx} </button> '
            hidden_divs += (
                f' <div  id="page{i}" style="display: block;" > {contents[idx]} </div> '
            )
        else:
            pagination += f' <button class="button is-small " href="  # " onclick=showPages{pages_uuid}("{i}")> {idx} </button> '
            hidden_divs += (
                f' <div  id="page{i}" style="display: none;"> {contents[idx]} </div> '
            )

    pagination += " </ul> "

    container = f"""
    {hidden_divs}

<div class="container is-fluid">


    {pagination}


</div>

    """

    js_paginate = f"""
    <script>

    function showPages{pages_uuid}(id){{
        var totalNumberOfPages = {number_pages};
        for(var i=0; i<=totalNumberOfPages.length; i++){{

            if (document.getElementById('page'+totalNumberOfPages[i])) {{

                document.getElementById('page'+totalNumberOfPages[i]).style.display='none';
            }}

        }}
            if (document.getElementById('page'+id)) {{

                document.getElementById('page'+id).style.display='block';
            }}
}};
    </script>
    """

    return f"""
    {container}

    {js_paginate}
    """


def bulma_message(header="Header", body="MessageBody"):
    return f"""
<article class="message is-info">
  <div class="message-header">
    <p>{header}</p>
    <button class="delete" aria-label="delete"></button>
  </div>
  <div class="message-body">
    {body}
  </div>
</article>
"""


def b64_div(data):
    return f'<img src="data:image/png;base64,{data}" />'


def bulma_input(placeholder="Text here", name=""):

    return f"""
<div class="field">
  <div class="control">
    <input class="input" type="text" placeholder="{placeholder}" name="{name}">
  </div>
</div>
"""


def flask_print(stderr=[], stdout=[]):
    import sys

    for i in stderr:
        print(i, file=sys.stderr)
    for i in stdout:
        print(i, file=sys.stdout)
    sys.stdout.flush()
    sys.stderr.flush()


def chartjs_multi(dataset, height=300, chart_type="horizontalBar"):
    chart_id = np.random.randint(0, 100)

    result = f"""
<canvas id="{chart_id}" height="{height}"></canvas>
<script>

var ctx = document.getElementById('{chart_id}').getContext('2d');
var chart = new Chart(ctx, {{
    // The type of chart we want to create
    type: '{chart_type}',

    // The data for our dataset
    {dataset} ,

    // Configuration options go here
    options: {{

    }}
}});

    </script>

    """

    return result


def chartjs_init_datasets(titles, labels, datasets, types, colors=None, fill=True):
    data = {}
    data["data"] = {}
    data["data"]["datasets"] = []
    for idx in range(len(titles)):
        temp = {}
        temp["label"] = titles[idx]
        temp["data"] = datasets[idx]
        temp["type"] = types[idx]
        if colors is None:
            temp["backgroundColor"] = "rgba(0, 0, 0, 0.1)"
        else:
            temp["backgroundColor"] = colors[idx]
        if fill is True:
            temp["fill"] = "true"
        else:
            temp["fill"] = "false"

        data["data"]["datasets"].append(temp)
    data["data"]["labels"] = labels

    return json.dumps(data)[:-1][1:]


def test_template():
    from flask import Flask

    app = Flask(__name__)

    @app.route("/<current_name>/")
    def template_test(current_name):
        log_buffer = []
        log_buffer.append("test this")
        client_log = """<script type="text/javascript">
        console.log("{}") </script>""".format(
            *log_buffer
        )

        result = f"""
               <!DOCTYPE html>
           <html>
           {bulma_header() + ' ' +  client_log}
            <body>


            </body>

            """
        return result

    if __name__ == "__main__":
        app.run(debug=True, port=80)

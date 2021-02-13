#
import json
import numpy as np
import uuid


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


def bulma_color_schemes():
    return [
        "brewer.YlGn3",
        "brewer.YlGn4",
        "brewer.YlGn5",
        "brewer.YlGn6",
        "brewer.YlGn7",
        "brewer.YlGn8",
        "brewer.YlGn9",
        "brewer.YlGnBu3",
        "brewer.YlGnBu4",
        "brewer.YlGnBu5",
        "brewer.YlGnBu6",
        "brewer.YlGnBu7",
        "brewer.YlGnBu8",
        "brewer.YlGnBu9",
        "brewer.GnBu3",
        "brewer.GnBu4",
        "brewer.GnBu5",
        "brewer.GnBu6",
        "brewer.GnBu7",
        "brewer.GnBu8",
        "brewer.GnBu9",
        "brewer.BuGn3",
        "brewer.BuGn4",
        "brewer.BuGn5",
        "brewer.BuGn6",
        "brewer.BuGn7",
        "brewer.BuGn8",
        "brewer.BuGn9",
        "brewer.PuBuGn3",
        "brewer.PuBuGn4",
        "brewer.PuBuGn5",
        "brewer.PuBuGn6",
        "brewer.PuBuGn7",
        "brewer.PuBuGn8",
        "brewer.PuBuGn9",
        "brewer.PuBu3",
        "brewer.PuBu4",
        "brewer.PuBu5",
        "brewer.PuBu6",
        "brewer.PuBu7",
        "brewer.PuBu8",
        "brewer.PuBu9",
        "brewer.BuPu3",
        "brewer.BuPu4",
        "brewer.BuPu5",
        "brewer.BuPu6",
        "brewer.BuPu7",
        "brewer.BuPu8",
        "brewer.BuPu9",
        "brewer.RdPu3",
        "brewer.RdPu4",
        "brewer.RdPu5",
        "brewer.RdPu6",
        "brewer.RdPu7",
        "brewer.RdPu8",
        "brewer.RdPu9",
        "brewer.PuRd3",
        "brewer.PuRd4",
        "brewer.PuRd5",
        "brewer.PuRd6",
        "brewer.PuRd7",
        "brewer.PuRd8",
        "brewer.PuRd9",
        "brewer.OrRd3",
        "brewer.OrRd4",
        "brewer.OrRd5",
        "brewer.OrRd6",
        "brewer.OrRd7",
        "brewer.OrRd8",
        "brewer.OrRd9",
        "brewer.YlOrRd3",
        "brewer.YlOrRd4",
        "brewer.YlOrRd5",
        "brewer.YlOrRd6",
        "brewer.YlOrRd7",
        "brewer.YlOrRd8",
        "brewer.YlOrRd9",
        "brewer.YlOrBr3",
        "brewer.YlOrBr4",
        "brewer.YlOrBr5",
        "brewer.YlOrBr6",
        "brewer.YlOrBr7",
        "brewer.YlOrBr8",
        "brewer.YlOrBr9",
        "brewer.Purples3",
        "brewer.Purples4",
        "brewer.Purples5",
        "brewer.Purples6",
        "brewer.Purples7",
        "brewer.Purples8",
        "brewer.Purples9",
        "brewer.Blues3",
        "brewer.Blues4",
        "brewer.Blues5",
        "brewer.Blues6",
        "brewer.Blues7",
        "brewer.Blues8",
        "brewer.Blues9",
        "brewer.Greens3",
        "brewer.Greens4",
        "brewer.Greens5",
        "brewer.Greens6",
        "brewer.Greens7",
        "brewer.Greens8",
        "brewer.Greens9",
        "brewer.Oranges3",
        "brewer.Oranges4",
        "brewer.Oranges5",
        "brewer.Oranges6",
        "brewer.Oranges7",
        "brewer.Oranges8",
        "brewer.Oranges9",
        "brewer.Reds3",
        "brewer.Reds4",
        "brewer.Reds5",
        "brewer.Reds6",
        "brewer.Reds7",
        "brewer.Reds8",
        "brewer.Reds9",
        "brewer.Greys3",
        "brewer.Greys4",
        "brewer.Greys5",
        "brewer.Greys6",
        "brewer.Greys7",
        "brewer.Greys8",
        "brewer.Greys9",
        "brewer.PuOr3",
        "brewer.PuOr4",
        "brewer.PuOr5",
        "brewer.PuOr6",
        "brewer.PuOr7",
        "brewer.PuOr8",
        "brewer.PuOr9",
        "brewer.PuOr10",
        "brewer.PuOr11",
        "brewer.BrBG3",
        "brewer.BrBG4",
        "brewer.BrBG5",
        "brewer.BrBG6",
        "brewer.BrBG7",
        "brewer.BrBG8",
        "brewer.BrBG9",
        "brewer.BrBG10",
        "brewer.BrBG11",
        "brewer.PRGn3",
        "brewer.PRGn4",
        "brewer.PRGn5",
        "brewer.PRGn6",
        "brewer.PRGn7",
        "brewer.PRGn8",
        "brewer.PRGn9",
        "brewer.PRGn10",
        "brewer.PRGn11",
        "brewer.PiYG3",
        "brewer.PiYG4",
        "brewer.PiYG5",
        "brewer.PiYG6",
        "brewer.PiYG7",
        "brewer.PiYG8",
        "brewer.PiYG9",
        "brewer.PiYG10",
        "brewer.PiYG11",
        "brewer.RdBu3",
        "brewer.RdBu4",
        "brewer.RdBu5",
        "brewer.RdBu6",
        "brewer.RdBu7",
        "brewer.RdBu8",
        "brewer.RdBu9",
        "brewer.RdBu10",
        "brewer.RdBu11",
        "brewer.RdGy3",
        "brewer.RdGy4",
        "brewer.RdGy5",
        "brewer.RdGy6",
        "brewer.RdGy7",
        "brewer.RdGy8",
        "brewer.RdGy9",
        "brewer.RdGy10",
        "brewer.RdGy11",
        "brewer.RdYlBu3",
        "brewer.RdYlBu4",
        "brewer.RdYlBu5",
        "brewer.RdYlBu6",
        "brewer.RdYlBu7",
        "brewer.RdYlBu8",
        "brewer.RdYlBu9",
        "brewer.RdYlBu10",
        "brewer.RdYlBu11",
        "brewer.Spectral3",
        "brewer.Spectral4",
        "brewer.Spectral5",
        "brewer.Spectral6",
        "brewer.Spectral7",
        "brewer.Spectral8",
        "brewer.Spectral9",
        "brewer.Spectral10",
        "brewer.Spectral11",
        "brewer.RdYlGn3",
        "brewer.RdYlGn4",
        "brewer.RdYlGn5",
        "brewer.RdYlGn6",
        "brewer.RdYlGn7",
        "brewer.RdYlGn8",
        "brewer.RdYlGn9",
        "brewer.RdYlGn10",
        "brewer.RdYlGn11",
        "brewer.Accent3",
        "brewer.Accent4",
        "brewer.Accent5",
        "brewer.Accent6",
        "brewer.Accent7",
        "brewer.Accent8",
        "brewer.DarkTwo3",
        "brewer.DarkTwo4",
        "brewer.DarkTwo5",
        "brewer.DarkTwo6",
        "brewer.DarkTwo7",
        "brewer.DarkTwo8",
        "brewer.Paired3",
        "brewer.Paired4",
        "brewer.Paired5",
        "brewer.Paired6",
        "brewer.Paired7",
        "brewer.Paired8",
        "brewer.Paired9",
        "brewer.Paired10",
        "brewer.Paired11",
        "brewer.Paired12",
        "brewer.PastelOne3",
        "brewer.PastelOne4",
        "brewer.PastelOne5",
        "brewer.PastelOne6",
        "brewer.PastelOne7",
        "brewer.PastelOne8",
        "brewer.PastelOne9",
        "brewer.PastelTwo3",
        "brewer.PastelTwo4",
        "brewer.PastelTwo5",
        "brewer.PastelTwo6",
        "brewer.PastelTwo7",
        "brewer.PastelTwo8",
        "brewer.SetOne3",
        "brewer.SetOne4",
        "brewer.SetOne5",
        "brewer.SetOne6",
        "brewer.SetOne7",
        "brewer.SetOne8",
        "brewer.SetOne9",
        "brewer.SetTwo3",
        "brewer.SetTwo4",
        "brewer.SetTwo5",
        "brewer.SetTwo6",
        "brewer.SetTwo7",
        "brewer.SetTwo8",
        "brewer.SetThree3",
        "brewer.SetThree4",
        "brewer.SetThree5",
        "brewer.SetThree6",
        "brewer.SetThree7",
        "brewer.SetThree8",
        "brewer.SetThree9",
        "brewer.SetThree10",
        "brewer.SetThree11",
        "brewer.SetThree12",
        "Microsoft Office",
        "office.Adjacency6",
        "office.Advantage6",
        "office.Angles6",
        "office.Apex6",
        "office.Apothecary6",
        "office.Aspect6",
        "office.Atlas6",
        "office.Austin6",
        "office.Badge6",
        "office.Banded6",
        "office.Basis6",
        "office.Berlin6",
        "office.BlackTie6",
        "office.Blue6",
        "office.BlueGreen6",
        "office.BlueII6",
        "office.BlueRed6",
        "office.BlueWarm6",
        "office.Breeze6",
        "office.Capital6",
        "office.Celestial6",
        "office.Circuit6",
        "office.Civic6",
        "office.Clarity6",
        "office.Codex6",
        "office.Composite6",
        "office.Concourse6",
        "office.Couture6",
        "office.Crop6",
        "office.Damask6",
        "office.Depth6",
        "office.Dividend6",
        "office.Droplet6",
        "office.Elemental6",
        "office.Equity6",
        "office.Essential6",
        "office.Excel16",
        "office.Executive6",
        "office.Exhibit6",
        "office.Expo6",
        "office.Facet6",
        "office.Feathered6",
        "office.Flow6",
        "office.Focus6",
        "office.Folio6",
        "office.Formal6",
        "office.Forte6",
        "office.Foundry6",
        "office.Frame6",
        "office.Gallery6",
        "office.Genesis6",
        "office.Grayscale6",
        "office.Green6",
        "office.GreenYellow6",
        "office.Grid6",
        "office.Habitat6",
        "office.Hardcover6",
        "office.Headlines6",
        "office.Horizon6",
        "office.Infusion6",
        "office.Inkwell6",
        "office.Inspiration6",
        "office.Integral6",
        "office.Ion6",
        "office.IonBoardroom6",
        "office.Kilter6",
        "office.Madison6",
        "office.MainEvent6",
        "office.Marquee6",
        "office.Median6",
        "office.Mesh6",
        "office.Metail6",
        "office.Metro6",
        "office.Metropolitan6",
        "office.Module6",
        "office.NewsPrint6",
        "office.Office6",
        "office.OfficeClassic6",
        "office.Opulent6",
        "office.Orange6",
        "office.OrangeRed6",
        "office.Orbit6",
        "office.Organic6",
        "office.Oriel6",
        "office.Origin6",
        "office.Paper6",
        "office.Parallax6",
        "office.Parcel6",
        "office.Perception6",
        "office.Perspective6",
        "office.Pixel6",
        "office.Plaza6",
        "office.Precedent6",
        "office.Pushpin6",
        "office.Quotable6",
        "office.Red6",
        "office.RedOrange6",
        "office.RedViolet6",
        "office.Retrospect6",
        "office.Revolution6",
        "office.Saddle6",
        "office.Savon6",
        "office.Sketchbook6",
        "office.Sky6",
        "office.Slate6",
        "office.Slice6",
        "office.Slipstream6",
        "office.SOHO6",
        "office.Solstice6",
        "office.Spectrum6",
        "office.Story6",
        "office.Studio6",
        "office.Summer6",
        "office.Technic6",
        "office.Thatch6",
        "office.Tradition6",
        "office.Travelogue6",
        "office.Trek6",
        "office.Twilight6",
        "office.Urban6",
        "office.UrbanPop6",
        "office.VaporTrail6",
        "office.Venture6",
        "office.Verve6",
        "office.View6",
        "office.Violet6",
        "office.VioletII6",
        "office.Waveform6",
        "office.Wisp6",
        "office.WoodType6",
        "office.Yellow6",
        "office.YellowOrange6",
        "Tableau",
        "tableau.Tableau10",
        "tableau.Tableau20",
        "tableau.ColorBlind10",
        "tableau.SeattleGrays5",
        "tableau.Traffic9",
        "tableau.MillerStone11",
        "tableau.SuperfishelStone10",
        "tableau.NurielStone9",
        "tableau.JewelBright9",
        "tableau.Summer8",
        "tableau.Winter10",
        "tableau.GreenOrangeTeal12",
        "tableau.RedBlueBrown12",
        "tableau.PurplePinkGray12",
        "tableau.HueCircle19",
        "tableau.OrangeBlue7",
        "tableau.RedGreen7",
        "tableau.GreenBlue7",
        "tableau.RedBlue7",
        "tableau.RedBlack7",
        "tableau.GoldPurple7",
        "tableau.RedGreenGold7",
        "tableau.SunsetSunrise7",
        "tableau.OrangeBlueWhite7",
        "tableau.RedGreenWhite7",
        "tableau.GreenBlueWhite7",
        "tableau.RedBlueWhite7",
        "tableau.RedBlackWhite7",
        "tableau.OrangeBlueLight7",
        "tableau.Temperature7",
        "tableau.BlueGreen7",
        "tableau.BlueLight7",
        "tableau.OrangeLight7",
        "tableau.Blue20",
        "tableau.Orange20",
        "tableau.Green20",
        "tableau.Red20",
        "tableau.Purple20",
        "tableau.Brown20",
        "tableau.Gray20",
        "tableau.GrayWarm20",
        "tableau.BlueTeal20",
        "tableau.OrangeGold20",
        "tableau.GreenGold20",
        "tableau.RedGold21",
        "tableau.Classic10",
        "tableau.ClassicMedium10",
        "tableau.ClassicLight10",
        "tableau.Classic20",
        "tableau.ClassicGray5",
        "tableau.ClassicColorBlind10",
        "tableau.ClassicTrafficLight9",
        "tableau.ClassicPurpleGray6",
        "tableau.ClassicPurpleGray12",
        "tableau.ClassicGreenOrange6",
        "tableau.ClassicGreenOrange12",
        "tableau.ClassicBlueRed6",
        "tableau.ClassicBlueRed12",
        "tableau.ClassicCyclic13",
        "tableau.ClassicGreen7",
        "tableau.ClassicGray13",
        "tableau.ClassicBlue7",
        "tableau.ClassicRed9",
        "tableau.ClassicOrange7",
        "tableau.ClassicAreaRed11",
        "tableau.ClassicAreaGreen11",
        "tableau.ClassicAreaBrown11",
        "tableau.ClassicRedGreen11",
        "tableau.ClassicRedBlue11",
        "tableau.ClassicRedBlack11",
        "tableau.ClassicAreaRedGreen21",
        "tableau.ClassicOrangeBlue13",
        "tableau.ClassicGreenBlue11",
        "tableau.ClassicRedWhiteGreen11",
        "tableau.ClassicRedWhiteBlack11",
        "tableau.ClassicOrangeWhiteBlue11",
        "tableau.ClassicRedWhiteBlackLight10",
        "tableau.ClassicOrangeWhiteBlueLight11",
        "tableau.ClassicRedWhiteGreenLight11",
        "tableau.ClassicRedGreenLight11",
    ]


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


def bulma_inside_then(function_call, content, cache=True):
    import uuid

    fun_id = uuid.uuid4().hex
    if cache == True:
        caching = f"""xx_{fun_id} = value;
console.log("{function_call.replace('"', "'")} cache: xx_{fun_id}")"""
    else:
        caching = ""
    return f"""
{function_call}.then((value) => {{
  {content}
  {caching}
}})
"""


def gen_chartjs_label(
    title,
    data,
    background="#D6E9C6",
    border_color=None,
    stack_id=None,
    chart_type="bar",
    fill=True,
    display=True,
    radius=None,
):
    if stack_id:
        stack_id = f"stack: '{stack_id}',"
    else:
        stack_id = ""

    if border_color:
        border_color = f"borderColor: '{border_color}',"
    else:
        border_color = ""

    if radius:
        radius = f"pointRadius: {radius},"
    else:
        radius = ""

    if display is True:
        display = " "
    else:
        display = f"""
hidden: true,
"""

    return f"""
{{
    label: '{title}',
    data: {data},
    backgroundColor: '{background}', // green
    {stack_id}
    {border_color}
    {display}
    {radius}
    type: '{chart_type}',
    fill: {json.dumps(fill)}
  }}
    """


def bulma_bar_chart(
    x,
    y,
    label,
    element_id,
    color_scheme=bulma_color_schemes()[30],
    var_id=uuid.uuid4().hex,
):
    return f"""

    cct_{var_id} = document.getElementById("{element_id}").getContext('2d')

    var bar_chart_{var_id} = new Chart(cct_{var_id}, {{
 type: 'bar',
    data: {{
        labels: {x},
        datasets: [{{
            label: '{label}',
            data: {y},
            borderWidth: 1
        }}]
    }},
    options: {{
        scales: {{
            yAxes: [{{
                ticks: {{
                    beginAtZero: true
                }}
            }}]
        }},
     plugins: {{
      colorschemes: {{
        scheme: '{color_scheme}'

      }}
    }}
}}

}}

);

    """


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

""" Utilities to use with beautifulsoup"""

from bs4 import BeautifulSoup
import requests


def make_soup(url, parser="html.parser"):
    """
    * type-def ::String :: PyObj -> BeautifulSoup
    * ---------------{Function}---------------
    * Take in a url and return beautifulsoup object . . .
    * ----------------{Params}----------------
    * : url    | String with the url to parse
    * : parser | beautifulsoup parser
    * ----------------{Returns}---------------
    * BeautifulSoup . . .
    """
    r = requests.get(url)
    contents = r.content
    soup = BeautifulSoup(contents, parser)
    return soup


def bs4_parse_table(tbody_tag):
    data = []
    rows = tbody_tag.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])  # Get rid of empty values
    return data


def bs4_parse_table_h(theader_tag):
    data = []
    rows = theader_tag.find_all("tr")
    for row in rows:
        cols = row.find_all("th")
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])  # Get rid of empty values
    return data


def bs4_find_class(soup, class_name):
    return soup.findAll("div", {"class": {class_name}})


def yahoo_current_time():
    import time

    return int(time.time())


def yahoo_splits_history(ticker):
    base_url = "https://finance.yahoo.com"
    period = f"period1=-252288000&period2={yahoo_current_time()}"
    interval = "interval=div%7Csplit"
    data_filter = "filter=split&frequency=1d"

    url = f"""{base_url}/quote/{ticker}/history?{period}&{interval}&{data_filter}"""

    soup = make_soup(url)
    historical_data = soup.find(id="Col1-1-HistoricalDataTable-Proxy")
    historical_table = historical_data.find("tbody")

    parsed_table = bs4_parse_table(historical_table)
    return parsed_table


def yahoo_institutional_data(ticker):
    base_url = "https://finance.yahoo.com"
    url = f"{base_url}/quote/{ticker}/holders?p={ticker}"
    soup = make_soup(url)

    summary_data = bs4_find_class(soup, "W(100%) Mb(20px)")

    summary_table = summary_data[0].find("tbody")
    summary_table = bs4_parse_table(summary_table)

    institutional_data = bs4_find_class(soup, "Mt(25px) Ovx(a) W(100%)")
    institutional_table = institutional_data[0].find("tbody")
    institutional_table = bs4_parse_table(institutional_table)

    mf_table = institutional_data[1].find("tbody")
    mf_table = bs4_parse_table(mf_table)

    return summary_table, institutional_table, mf_table


def yahoo_financial_data(ticker):
    base_url = "https://finance.yahoo.com"

    url = f"{base_url}/quote/{ticker}/analysis?p={ticker}"

    soup = make_soup(url)

    summary_data = bs4_find_class(
        soup,
        "Pos(r) Bgc($bg-content) Bgc($lv2BgColor)! Miw(1007px) Maw(1260px) tablet_Miw(600px)--noRightRail Bxz(bb) Bdstartc(t) Bdstartw(20px) Bdendc(t) Bdends(s) Bdendw(20px) Bdstarts(s) Mx(a)",
    )

    summary_data = summary_data[0].find(id="YDC-Col1")

    summary_data = summary_data.find(id="Main")

    r_n = int(summary_data.get_attribute_list("data-reactid")[0]) + 1

    summary_data = summary_data.find("div", attrs={"data-reactid": r_n})

    summary_data = summary_data.find(id="mrt-node-Col1-0-AnalystLeafPage")

    summary_data = summary_data.find(id="Col1-0-AnalystLeafPage-Proxy")

    r_n = int(summary_data.get_attribute_list("data-reactid")[0]) + 1
    summary_data = summary_data.find("section", attrs={"data-reactid": r_n})

    summary_table = summary_data.findAll("table")
    summary_table_result = []
    for i in range(len(summary_table)):
        summary_table_result.append(
            (
                bs4_parse_table_h(summary_table[i].find("thead")),
                bs4_parse_table(summary_table[i].find("tbody")),
            )
        )
    return summary_table_result


def fetch_short_interest_tables(ticker):
    base_url = "https://fintel.io/ss/us"
    url = f"{base_url}/s?t={ticker}"
    soup = make_soup(url)

    summary_table = soup.find("tbody")

    try:
        short_summary = bs4_parse_table(summary_table)
    except Exception:
        short_summary = ""
    summary_data = bs4_find_class(soup, "card mt-3")
    try:
        short_interest = summary_data[0].findAll("table")
        short_interest_result = []
        for i in range(len(short_interest)):
            short_interest_result.append(
                (
                    bs4_parse_table_h(short_interest[i].find("thead")),
                    bs4_parse_table(short_interest[i].find("tbody")),
                )
            )
        short_interest_result = short_interest_result[0]
    except Exception:
        short_interest_result = ""

    try:
        short_volume = summary_data[1].findAll("table")
        short_volume_result = []
        for i in range(len(short_volume)):
            short_volume_result.append(
                (
                    bs4_parse_table_h(short_volume[i].find("thead")),
                    bs4_parse_table(short_volume[i].find("tbody")),
                )
            )
        short_volume_result = short_volume_result[0]
    except Exception:
        short_volume_result = ""

    return short_summary, short_interest_result, short_volume_result


def form_whispers_summary(ticker):
    base_url = "https://formwhispers.com"
    url = f"{base_url}/s?t={ticker}"
    soup = make_soup(url)

    summary_data = bs4_find_class(soup, "columns is-centered")
    summary_data = bs4_find_class(summary_data[1], "column is-8")

    description = summary_data[0].findAll("p")[2].text.strip()

    summary_table = soup.find("tbody")
    summary_table = bs4_parse_table(summary_table)
    return description, summary_table


def marketbeat_short_data(ticker):
    base_url = "https://www.marketbeat.com/stocks/NYSE/"
    url = f"{base_url}/{ticker}/short-interest/"
    soup = make_soup(url)
    data = bs4_parse_table(soup)

    short_interest_data = data[:12]
    short_interest_history = data[12:][:-8]
    return short_interest_data, short_interest_history


def snp_50():
    url = "http://www.eoddata.com/stocklist/NASDAQ.htm?AspxAutoDetectCookieSupport=1"
    soup = make_soup(url)
    # summary_data = soup.find("div",{"id":"ctl00_cph1_pnl1"})
    summary_data = bs4_find_class(soup, "cb")
    summary_data = bs4_parse_table(summary_data[0])

    return summary_data


def yahoo_price_history(ticker):
    base_url = "https://finance.yahoo.com"
    period = f"period1=-252374400&period2={yahoo_current_time()}"
    interval = "interval=1d"
    data_filter = "filter=history&frequency=1d"
    url = f"{base_url}/quote/{ticker}/history?{period}&{interval}&{data_filter}&includeAdjustedClose=true"
    soup = make_soup(url)

    summary_data = bs4_find_class(
        soup,
        "Pb(10px) Ovx(a) W(100%)",
    )

    summary_data = bs4_parse_table(summary_data[0])

    return summary_data[1:][:-1]


def yahoo_dividends_history(ticker):
    base_url = "https://finance.yahoo.com"
    period = f"period1=-252288000&period2={yahoo_current_time()}"
    interval = "interval=div%7Csplit"
    data_filter = "filter=div&frequency=1d"

    url = f"{base_url}/quote/{ticker}/history?{period}&{interval}&{data_filter}"

    try:
        soup = make_soup(url)
        historical_data = soup.find(id="Col1-1-HistoricalDataTable-Proxy")
        historical_table = historical_data.find("tbody")

        parsed_table = bs4_parse_table(historical_table)
    except Exception:
        parsed_table = ""

    return parsed_table


def yahoo_find_summary(ticker):
    # ticker = "PSXP"
    base_url = "https://finance.yahoo.com/quote"
    url = f"{base_url}/{ticker}"
    soup = make_soup(url)
    summary_data = soup.findAll("table")
    current_info_summary = bs4_parse_table(summary_data[0])
    fin_info_summary = bs4_parse_table(summary_data[1])
    return current_info_summary, fin_info_summary


def render_js(url):
    session = HTMLSession()
    r = session.get(url)
    r.html.render()
    return r.content


def quick_html_inspect(response, save_to_file="response.html", snippet_length=500):
    from bs4 import BeautifulSoup

    if response.status_code == 200:
        print("\033[32m[INFO] HTTP Response: 200 OK\033[0m")
        try:
            html_content = response.content.decode("utf-8")
        except UnicodeDecodeError:
            print("\033[31m[ERROR] Could not decode the response content.\033[0m")
            return

        print("\033[35m[HTML Snippet]\033[0m")
        print(html_content[:snippet_length])

        if save_to_file:
            with open(save_to_file, "w", encoding="utf-8") as file:
                file.write(html_content)
            print(f"\033[34m[INFO] Full HTML saved to '{save_to_file}'.\033[0m")

        soup = BeautifulSoup(html_content, "html.parser")
        print(
            "\033[36m[INFO] Page Title:\033[0m",
            soup.title.string if soup.title else "No title found",
        )
        print(
            "\033[36m[INFO] First 500 characters of body:\033[0m",
            soup.body.text[:500] if soup.body else "No body found",
        )
    else:
        print(f"\033[31m[ERROR] HTTP Response: {response.status_code}\033[0m")

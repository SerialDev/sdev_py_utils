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

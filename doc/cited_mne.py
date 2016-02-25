"""Parse google scholar -> rst for MNE citations."""

# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
# License : BSD 3-clause

# Parts of this code were copied from google_scholar_parser
# (https://github.com/carlosp420/google_scholar_parser)

import re
import time
import random
import requests

from BeautifulSoup import BeautifulSoup
from mne.externals.tempita import Template


UA = ('Mozilla/5.0 (X11; U; FreeBSD i386; en-US; rv:1.9.2.9) '
      'Gecko/20100913 Firefox/3.6.9')

# ##### Templates for ciations #####
html = (u""".. _cited

Citations
=========

Papers citing MNE as extracted from Google Scholar (on %s).
""")

cite_template = Template(u"""
{{paper}}

{{for ii, author in enumerate(authors)}}
{{ii + 1}}. `{{titles[ii]}} <{{links[ii]}}>`_. {{author}}.
{{endfor}}

""")


def parse_soup_page(soup):
    """Parse the page using BeautifulSoup.

    Parameters
    ----------
    soup : instance of BeautifulSoup
        The page to be parsed.

    Returns
    -------
    titles : list
        The article titles.
    authors : list
        The name of the authors.
    links : list
        Hyperlinks to the articles.
    """
    titles, authors, links = list(), list(), list()
    for div in soup.findAll('div'):
        if div.name == "div" and div.get('class') == "gs_ri":
            links.append(div.a['href'])
            div_pub = div.findAll('div')
            for d in div_pub:
                if d.name == 'div' and d.get('class') == 'gs_a':
                    authors.append(d.text)
            titles.append(div.a.text)
    return titles, authors, links


def get_total_citations(soup):
    """Get total citations."""
    results = soup.find('div', attrs={'id': 'gs_ab_md'}).contents[0]
    matches = re.search("About\s(\d+)\s", results)
    if matches:
        hits = matches.groups()[0]
        return hits


def _get_soup(url, backend='selenium'):
    """Get BeautifulSoup object from url.

    Parameters
    ----------
    url : str
        The url to fetch.
    backend : 'selenium' | 'requests'
        Use selenium by default because google can ask for captcha. For
        'selenium' backend Firefox must be installed.

    Returns
    -------
    soup : instance of BeautifulSoup
        The soup page from the url.
    """
    if backend == 'requests':
        req = requests.get(url, headers={'User-Agent': UA})
        html_doc = req.text
        soup = BeautifulSoup(html_doc)
        if soup.find('div', attrs={'id': 'gs_ab_md'}) is None:
            print('Falling back on to selenium backend due to captcha.')
            backend = 'selenium'

    if backend == 'selenium':
        from selenium import webdriver
        import selenium.webdriver.support.ui as ui

        driver = webdriver.Firefox()
        # give enough time to solve captcha.
        wait = ui.WebDriverWait(driver, 200)

        driver.get(url)
        wait.until(lambda driver: driver.find_elements_by_id('gs_ab_md'))

        html_doc = driver.page_source
        soup = BeautifulSoup(html_doc)
        driver.close()

    return soup


def get_citing_articles(cites_url, backend):
    """Get the citing articles.

    Parameters
    ----------
    cites_url: str
        A citation url from Google Scholar.
    backend : 'selenium' | 'requests'
        Use selenium by default because google can ask for captcha. For
        'selenium' backend Firefox must be installed.


    Returns
    -------
    titles : list
        The article titles.
    authors : list
        The name of the authors.
    links : list
        Hyperlinks to the articles.
    """
    n = random.random() * 5
    time.sleep(n)
    print("\nSleeping: {0} seconds".format(n))

    # GS seems to allow only 20 hits per page!
    cites_url += "&num=20"
    soup = _get_soup(cites_url, backend=backend)
    hits = get_total_citations(soup)
    print("Got a total of {0} citations".format(hits))

    hits = int(hits)
    index = 0
    titles, authors, links = list(), list(), list()
    while hits > 1:
        n = random.random() * 2
        time.sleep(n)
        if index > 0:
            url = cites_url + "&start=" + str(index)
        else:
            url = cites_url
        index += 20
        hits -= 20
        print("{0} more citations to process".format(hits))
        soup = soup = _get_soup(url, backend=backend)
        title, author, link = parse_soup_page(soup)
        for this_title, this_author, this_link in zip(title, author, link):
            titles.append(this_title)
            authors.append(this_author)
            links.append(this_link)

    return titles, authors, links

if __name__ == '__main__':
    backend = 'requests'
    random.seed()
    gen_date = time.strftime("%B %d, %Y")
    html = html % gen_date

    url_tails = ['1521584321377182930', '12188330066413208874']
    papers = ['MEG and EEG data analysis with MNE-Python',
              'MNE software for processing MEG and EEG data']

    for url_tail, paper in zip(url_tails, papers):
        titles, authors, links = get_citing_articles(
            'https://scholar.google.co.in/scholar?cites=%s'
            % url_tail, backend=backend)

        titles = [title.encode('utf8') for title in titles]
        authors = [author.encode('utf8') for author in authors]

        paper = '\n'.join([paper, '-' * len(paper)])

        # create rst & cleanup
        this_html = cite_template.substitute(paper=paper, titles=titles,
                                             authors=authors, links=links)
        this_html = this_html.replace('&hellip;', '...')

        html += this_html

    # output an rst file
    with open('cited.rst', 'w') as f:
        f.write(html.encode('utf8'))

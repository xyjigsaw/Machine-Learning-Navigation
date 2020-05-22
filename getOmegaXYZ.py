# Name: getOmegaXYZ
# Author: Reacubeth
# Time: 2020-05-22 21:22
# Mail: noverfitting@gmail.com
# Site: www.omegaxyz.com
# *_*coding:utf-8 *_*

import requests
from bs4 import BeautifulSoup
import requests
import urllib3
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
urllib3.disable_warnings()

headers = {
    'User-Agent': 'OmegaXYZ Spider',
}


def get(url: str):
    rep = requests.get(url, verify=False, headers=headers)
    rep.encoding = 'utf-8'
    return rep.text


def solver(url: str):
    res = get(url)
    soup = BeautifulSoup(res, features="lxml")
    posts = []
    for item in soup.select("h2.entry-title"):
        link = item.select_one("a")
        # posts.append({'title': link.get_text(), 'link': link.get("href")})
        posts.append('**[' + link.get_text() + '](' + link.get("href") + ')**')
    try:
        next_page = soup.select("a.next")[0].get("href")
    except IndexError:
        next_page = ''
    return posts, next_page


if __name__ == '__main__':
    next_url = "https://www.omegaxyz.com/category/tech/machine-learning/"
    while next_url:
        ls, next_url = solver(next_url)
        for i in ls:
            print(i)


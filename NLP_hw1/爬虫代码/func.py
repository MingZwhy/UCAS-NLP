# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/19 10:31
@Auth ： 谷朝阳
@File ：func.py
"""

import requests
from bs4 import BeautifulSoup
import unicodedata
import re


def geturl():
    url = "https://www.xqb5200.com/3_3248/"
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    req = requests.get(url=url, headers=header)
    req.encoding = "gbk"
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find("div", id="list")
    chapters = texts.find_all("a")
    words = []
    print(chapters)
    for chapter in chapters:
        name = chapter.string
        url1 = url + chapter.get("href")
        word = [url1, name]
        words.append(word)
    return words


if __name__ == '__main__':
    target = geturl()
    print(target)
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    for tar in target:
        req = requests.get(url=tar[0], headers=header)
        req.encoding = 'gbk'
        html = req.text
        bes = BeautifulSoup(html, "lxml")
        texts = bes.find("div", id="content")
        texts_list = texts.text.split("\xa0" * 4)
        try:
            with open("F:/novels/Dou3/" + tar[1] + ".txt", "w") as file:
                for line in texts_list:
                    line.replace(u'\xa0', '')
                    file.write(line + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue

# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 20:35
@Auth ： 谷朝阳
@File ：func5.py
"""

import requests
from bs4 import BeautifulSoup
import unicodedata
import re


def geturl():
    url = "http://cpc.people.com.cn/GB/67481/444924/444926/index2.html"
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    req = requests.get(url=url, headers=header)
    req.encoding = "gbk"
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find(attrs={'class': 'list'})
    # print(texts)
    chapters = texts.find_all("a")
    words = []
    # herf = str(chapters[0].a.get('href'))
    print(chapters)
    for chapter in chapters:
        name = chapter.text[:10]
        url1 = url + str(chapter.get('href'))
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
        texts = bes.find(attrs={'class': 'show_text'})
        if not texts:
            continue
        texts_list = texts.text.split("\xa0" * 4)
        try:
            with open("F:/novels/XinHua/15/" + str(tar[1]).replace("|","") + ".txt", "w") as file:
                for line in texts_list:
                    line = line.replace(" ", "").replace("\n", "")
                    line = line.replace(u'\xa0', '')
                    file.write(line + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue

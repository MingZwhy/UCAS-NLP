# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/19 10:31
@Auth ： 谷朝阳
@File ：func2.py
"""

import requests
from bs4 import BeautifulSoup
import unicodedata
import re


def geturl():
    url = "http://opinion.people.com.cn/GB/51854/index2.html"
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    req = requests.get(url=url, headers=header)
    req.encoding = "gbk"
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    # texts = bes.find("ul")
    texts = bes.find_all("li")
    chapters = bes.find_all("li")
    words = []
    # herf = str(chapters[0].a.get('href'))

    for chapter in chapters:
        name = chapter.text
        url1 = url + str(chapter.a.get('href'))
        word = [url1, name]
        words.append(word)
    return words


if __name__ == '__main__':
    target = geturl()
    # print(target)

    # url = 'http://opinion.people.com.cn/GB/223228/index1.html/n1/2023/0326/c223228-32651297.html'

    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    # req = requests.get(url=url, headers=header)
    # req.encoding = 'gbk'
    # html = req.text
    # bes = BeautifulSoup(html, "lxml")
    # body = bes.find(attrs={'class': 'rm_txt_con cf'})
    # texts = body.find_all("p")
    # print(len(body.text.split("\xa0" * 4)))
    for tar in target:
        req = requests.get(url=tar[0], headers=header)
        req.encoding = 'gbk'
        html = req.text
        bes = BeautifulSoup(html, "lxml")
        texts = bes.find(attrs={'class': 'rm_txt_con cf'})
        texts_list = texts.text.split("\xa0" * 4)
        try:
            with open("F:/novels/XinHua/12/" + tar[1].split(' ')[0].replace('?','') + ".txt", "w") as file:
                for line in texts_list:
                    line = line.replace(" ", "").replace("\n", "")
                    line = line.replace(u'\xa0', '')
                    file.write(line + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue

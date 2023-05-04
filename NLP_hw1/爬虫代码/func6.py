# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 21:26
@Auth ： 谷朝阳
@File ：func6.py
"""

import requests
from bs4 import BeautifulSoup
import unicodedata
import re
from tqdm import tqdm

def geturl(page):
    url = "http://jhsjk.people.cn/result?searchArea=0&keywords=&isFuzzy=0"
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    params = {
        'page': page
    }
    req = requests.get(url=url, params=params, headers=header)
    req.encoding = "utf-8"
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find(id="news_list")
    # print(texts)
    chapters = texts.find_all("a")
    words = []
    # herf = str(chapters[0].a.get('href'))
    # print(chapters)
    for chapter in chapters:
        name = chapter.text[:20]
        url1 = 'http://jhsjk.people.cn/' + str(chapter.get('href'))
        word = [url1, name]
        words.append(word)
    return words


if __name__ == '__main__':
    for index in tqdm(range(800,12970)):
        target = geturl(index)
        # print(target)
        header = {"User-Agent":
                      "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                      "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
        for tar in target:
            req = requests.get(url=tar[0], headers=header)
            req.encoding = 'utf-8'
            html = req.text
            bes = BeautifulSoup(html, "lxml")
            texts = bes.find(attrs={'class': 'd2txt_con clearfix'})
            if not texts:
                continue
            texts_list = texts.text.split("\xa0" * 4)
            try:
                with open("F:/novels/XinHua/17/" + str(tar[1]).replace("|", "").replace("\"", "").replace(" ", "").replace("\n", "").replace("，", "").replace("/", "") + ".txt", "w") as file:
                    for line in texts_list:
                        line = line.replace(" ", "").replace("\n", "")
                        line = line.replace(u'\xa0', '')
                        file.write(line + "\n")
            except UnicodeEncodeError as e:
                print(f"UnicodeEncodeError: {e}")
                continue

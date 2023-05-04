# -*- coding: utf-8 -*-
"""
@Time ： 2023/3/26 12:13
@Auth ： 谷朝阳
@File ：func3.py
"""
import requests
from bs4 import BeautifulSoup
import unicodedata
import re


def geturl():
    url = "http://jhsjk.people.cn/result?type=102"
    header = {"User-Agent":
                  "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.25 "
                  "Safari/537.36 Core/1.70.3861.400 QQBrowser/10.7.4313.400"}
    req = requests.get(url=url, headers=header)
    req.encoding = "utf-8"
    html = req.text
    bes = BeautifulSoup(html, "lxml")
    texts = bes.find(id="news_list")
    # print(texts)
    chapters = texts.find_all("a")
    words = []
    # herf = str(chapters[0].a.get('href'))
    # print(chapters[0].get('href'))
    for chapter in chapters:
        name = chapter.text
        url1 = url + str(chapter.get('href'))
        word = [url1, name]
        words.append(word)
    return words


if __name__ == '__main__':
    target = geturl()
    print(target)
    # dict = ['32645371', '32645364','32643813','32645366','32643805','32642944','32642860','32642276','32642277','32642278']
    # dict = ['3264'+"{:04d}".format(i) for i in range(10000)]
    with open('F:/novels/XinHua/dict.txt', 'r') as f:
        dict = [line.strip() for line in f.readlines()]
    print(dict)
    # assert(0)
    target = [['http://jhsjk.people.cn/article/' + dict[i], i] for i in range(len(dict))]
    # url = 'http://opinion.people.com.cn/GB/223228/index1.html/n1/2023/0326/c223228-32651297.html'
    print(target)
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
        req.encoding = 'utf-8'
        html = req.text
        bes = BeautifulSoup(html, "lxml")
        texts = bes.find(attrs={'class': 'd2txt_con clearfix'})
        if not texts:
            continue
        texts_list = texts.text.split("\xa0" * 4)
        try:
            with open("F:/novels/XinHua/13/" + str(tar[1]) + ".txt", "w") as file:
                for line in texts_list:
                    line = line.replace(" ", "").replace("\n", "")
                    line = line.replace(u'\xa0', '')
                    file.write(line + "\n")
        except UnicodeEncodeError as e:
            print(f"UnicodeEncodeError: {e}")
            continue

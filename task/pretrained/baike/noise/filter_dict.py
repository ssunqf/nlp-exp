#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from collections import deque
from typing import List
from requests_html import HTMLSession


def make_url(text: str) -> str:
    return 'http://www.baidu.com/s?q1=&q2=%s&q3=&q4=&gpc=stf&ft=&q5=&q6=&tn=baiduadv' % text


def get_freq(words: List[str]):
    visited = []
    Q = deque(words)
    session = HTMLSession()

    while len(Q) > 0:
        word = Q.pop()
        response = session.get(make_url(word))
        if response.status_code == 200:
            visited.append(word)
            info = response.html.find('.nums_text', first=True)
            print(info.html)

get_freq(['你好'])
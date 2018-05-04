#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


import sys
import re

"""
  fix 'The provision of $310,300 is requested for:\t29. 310 300美元的编列经费用于' ->
      'The provision of $310,300 is requested for:\t29. 310300美元的编列经费用于'
"""
number_re = re.compile('(([0-9]{1,3})( [0-9]{3})+)([^0-9]|$)')


def fix_zh_number(en_text, zh_text):
    matches = number_re.findall(zh_text)
    for match in matches:
        old_format = match[0]
        en_format = match[0].replace(' ', ',')
        if en.find(en_format) != -1:
            zh_text = zh_text.replace(old_format, en_format)

    return en_text, zh_text



for line in sys.stdin:
    en, zh = line.strip().split('\t')

    en, zh = fix_zh_number(en, zh)

    print(en + '\t' + zh)



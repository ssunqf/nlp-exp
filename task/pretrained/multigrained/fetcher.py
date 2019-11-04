#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from urllib import request

from urllib.parse import quote

import string


text = '非公有制经济'

model = 'FFN'

url_base = 'http://101.132.166.249:5000/api?'

url = url_base + 'text='+text+'&model='+model

url=quote(url,safe=string.printable)


response=request.urlopen(url).read()

response=response.decode('utf-8')

print(response)
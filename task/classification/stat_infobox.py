#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from .utils import dbConnect, Entity
import json
from typing import Mapping
from bs4 import BeautifulSoup
from collections import defaultdict
import asyncio
from tqdm import tqdm


async def extract():
    pro_names = defaultdict(int)
    properties = defaultdict(int)

    db = await dbConnect()

    with tqdm(desc='processing') as t:
        async with db.transaction():
            async for record in db.cursor('select url, knowledge from baike_knowledge'):
                t.update(1)
                url = record['url']
                knowledge = json.loads(record['knowledge'])

                entity = Entity(knowledge, dict())
                for k, values in entity.infobox().items():
                    pro_names[k] += 1
                    for v in values:
                        v = BeautifulSoup(v, 'html.parser').getText().strip()
                        properties[k+'##'+v] += 1

    with open('state.pro_name', 'w') as f:
        f.writelines('%s\t%d\n' % (k, v)
                     for k, v in sorted(pro_names.items(), key=lambda v: v[1], reverse=True))

    with open('state.property', 'w') as f:
        f.writelines('%s\t%d\n' % (k, v)
                     for k, v in sorted(properties.items(), key=lambda v: v[1], reverse=True))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(extract())
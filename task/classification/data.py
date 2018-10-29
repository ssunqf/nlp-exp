#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import re
import json
from typing import List, Optional, Dict
from .utils import dbConnect, Entity
import asyncio
from .vectorize import BOWVectorizer
import sys
import numpy as np
from .config import config
from tqdm import tqdm


async def extract(vectorizer):
    db = await dbConnect()
    async with db.transaction():
        async for record in db.cursor('select url, knowledge, label from label_entity order by keyword'):
            url = record['url']
            knowledge = json.loads(record['knowledge'])
            label = json.loads(record['label'])

            entity = Entity(knowledge, label)
            if len(entity.labels()) > 0:
                summary = vectorizer.feature(entity.summary())
                tags = vectorizer.feature(entity.open_tags())
                keywords = vectorizer.feature(entity.keywords())
                infobox = vectorizer.feature(entity.infobox())
                type = vectorizer.feature(entity.type())

                feature = np.concatenate((summary, tags, keywords, infobox, type)).round(6)
                labels = entity.labels()
                print(url + '\t' + ' '.join(labels) + '\t' + ' '.join([str(int(i*1e6)/1e6) for i in feature]))


if __name__ == '__main__':

    vectorizer = BOWVectorizer(config.word_embed_path, config.mode)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(extract(vectorizer))



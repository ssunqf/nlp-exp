#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import re
import json
from typing import List, Optional, Dict
from .utils import dbConnect, dictWalk, html2text, splitter
import asyncio
from .vectorize import BOWVectorizer
import sys
import numpy as np
from .config import config


class Entity:
    def __init__(self, knowledge: Dict, label: Dict):
        self.knowledge = knowledge
        self.label = label

    def title(self):
        return dictWalk(self.knowledge, ['title'])

    def type(self) -> Optional[str]:
        title = dictWalk(self.knowledge, ['attrs', 'lemma_title'])
        if title:
            res = re.findall('(?<=（).*(?=）_百度百科)', title)
            if len(res) > 0:
                return res[0]
        return None

    def summary(self) -> List[str]:
        paras = html2text(dictWalk(self.knowledge, ['attrs', 'lemma_summary']))
        return [t for p in paras for t in splitter(p) if len(t) > 0] if paras and len(paras) > 0 else []

    def infobox(self) -> Dict[str, List[str]]:
        box = html2text(dictWalk(self.knowledge, ['attrs', 'infobox']))
        return box if box else {}

    def open_tags(self) -> List[str]:
        tags = html2text(dictWalk(self.knowledge, ['attrs', 'open_tags']))
        return tags if tags else []

    def keywords(self) -> List[str]:
        t = dictWalk(self.knowledge, ['attrs', 'lemma_title'])
        text = dictWalk(self.knowledge, ['keywords'])
        raws = [word.replace('###', ' ').replace(t, '') if t else word.replace('###', ' ')
                for word in re.sub('(?<=[A-Za-z.])[ ]+(?=[A-Za-z.])', '###', text if text else '').split()]
        return [i for i in raws if len(i) > 0]

    def labels(self) -> List[str]:

        def _filter():
            for k, v in self.label.items():
                if 'flag' in v and v['flag'] == 'manual' and 'mask' in v and v['mask'] == 1:
                    yield k

        return list(_filter())


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

                feature = np.concatenate((summary, tags, keywords, infobox, type))
                labels = entity.labels()
                print(url + '\t' + ' '.join(labels) + '\t' + ' '.join([str(int(i*1e6)/1e6) for i in feature]))

if __name__ == '__main__':

    vectorizer = BOWVectorizer(config.word_embed_path, config.mode)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(extract(vectorizer))



#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import re
import json
from typing import List, Optional, Dict
from .utils import dbConnect, dictWalk, html2text, splitter
import asyncio
from .vectorize import BOWVectorizer
import sys
import torch
import numpy as np
from .data import Entity
from .model import Classifier
from .config import config


async def predict(vectorizer, model, label2id: dict):
    db = await dbConnect()
    async with db.transaction():
        async for record in db.cursor('select url, knowledge, label from label_entity'):
            url = record['url']
            knowledge = json.loads(record['knowledge'])
            label = json.loads(record['label'])

            entity = Entity(knowledge, label)
            labels = entity.labels()
            if len(labels) > 0:
                summary = vectorizer.feature(entity.summary())
                tags = vectorizer.feature(entity.open_tags())
                keywords = vectorizer.feature(entity.keywords())
                infobox = vectorizer.feature(entity.infobox())
                type = vectorizer.feature(entity.type())

                feature = torch.from_numpy(np.concatenate((summary, tags, keywords, infobox, type))).unsqueeze(0).cuda()
                predict = model(feature).squeeze(0).cpu()

                wrongs = ['%s: %0.3f' % (label, predict[id])
                          for label, id in label2id.items()
                          if (predict[id] > 0.5 and label not in labels) or (predict[id] < 0.5 and label in labels)]
                if len(wrongs) > 0:
                    print(url)
                    print(knowledge)
                    print(labels)
                    print('\t'.join(wrongs))


if __name__ == '__main__':
    vectorizer = BOWVectorizer(config.word_embed_path, config.mode)
    print('load word embedding.')
    with open(config.label_dict) as f:
        labels = [l.strip() for l in f.readlines() if len(l.strip()) > 0 and not l.startswith('#')]
        label2id = dict((l, id) for id, l in enumerate(labels))
    model = torch.load(sys.argv[1])
    print('load model.')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(predict(vectorizer, model, label2id))

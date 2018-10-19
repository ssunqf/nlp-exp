#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import re
import json
from typing import List, Optional, Dict
from .utils import dbConnect, Entity
import asyncio
from .vectorize import BOWVectorizer
import sys
import torch
import numpy as np

from .model import Classifier
from .config import config
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir=config.summary_dir)

async def predict(vectorizer, model, label2id: dict):
    db = await dbConnect()
    async with db.transaction():
        batch_feature = []
        batch_label = []
        batch_id = 0
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

                feature = torch.from_numpy(np.concatenate((summary, tags, keywords, infobox, type))).unsqueeze(0)
                predict = model(feature).squeeze(0)

                batch_feature.append(feature)
                batch_label.append(labels[0] + '\t' + url)

                if len(batch_feature) > 0 and len(batch_feature) == 2000:
                    batch_feature = torch.cat(batch_feature)
                    writer.add_embedding(batch_feature, metadata=batch_label, global_step=batch_id)

                    batch_feature = []
                    batch_label = []
                    batch_id += 1

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
    writer = SummaryWriter(log_dir=config.summary_dir)
    with open(config.label_dict) as f:
        labels = [l.strip() for l in f.readlines() if len(l.strip()) > 0 and not l.startswith('#')]
        label2id = dict((l, id) for id, l in enumerate(labels))
    model = torch.load(sys.argv[1])
    print('load model.')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(predict(vectorizer, model, label2id))

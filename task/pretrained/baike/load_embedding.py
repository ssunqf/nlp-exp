
#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import argparse
import torch
from typing import List
from os import makedirs, path

from .train import Config, Trainer


def save(path, vocab: List[str], weight: torch.Tensor):
    with open(path, 'w') as output:
        output.write('%d %d\n' % (len(vocab), weight.size(1)))
        for id, str in enumerate(vocab):
            vector = weight[id].tolist()
            output.write(str + ' ' + ' '.join('%f' % i for i in vector) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess baike corpus and save vocabulary')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--output', type=str, help='ouput dir')

    args = parser.parse_args()

    config = Config()

    TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL = Trainer.load_voc(config)
    model = Trainer.load_elmo_model(config, TEXT, KEY_LABEL, ATTR_LABEL, SUB_LABEL, ENTITY_LABEL)
    states = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(states['model'])

    makedirs(args.output, exist_ok=True)
    save(path.join(args.output, 'word.vec'), model.text_voc.itos, model.embedding.weight)

    for classifier in model.label_classifiers:
        name, weight, vocab = classifier.named_embeddings()
        save(path.join(args.output, classifier.name + '.vec'), vocab, weight)

    torch.save(model.encoder, path.join(args.output, 'encoder.pk'))

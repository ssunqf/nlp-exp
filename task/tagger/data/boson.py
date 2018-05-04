#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import sys
from task.util import utils

if __name__ == '__main__':
    with open(sys.argv[1]) as reader, open(sys.argv[2], 'w') as writer:
        for line in reader:
            for sentence in line.split('\\n\\n'):
                chars = []
                tags = []
                for left in sentence.split('{{'):
                    for field in left.split('}}'):
                        type = None
                        text = field
                        if field.startswith('person_name:'):
                            type = 'PERSON'
                            text = field[len('person_name:'):]
                        elif field.startswith('company_name:'):
                            type = 'ORGANIZATION'
                            text = field[len('company_name:'):]
                        elif field.startswith('org_name:'):
                            type = 'ORGANIZATION'
                            text = field[len('org_name:'):]
                        elif field.startswith('location:'):
                            type = 'LOCATION'
                            text = field[len('location:'):]
                        elif field.startswith('product_name:'):
                            type = None
                            text = field[len('product_name:'):]
                        elif field.startswith('time:'):
                            type = 'TIME'
                            text = field[len('time:'):]

                        t_chars = [word for t, word in utils.replace_entity(text.strip())]
                        if type:
                            t_tags = ['S_'+type] if len(t_chars) == 1 else \
                                ['B_'+type] + ['M_'+type] * (len(t_chars) - 2) + ['E_'+type]
                        else:
                            t_tags = ['O'] * len(t_chars)

                        chars.extend(t_chars)
                        tags.extend(t_tags)

                writer.write(' '.join([char + '#' + tag for char, tag in zip(chars, tags)]) + '\n')



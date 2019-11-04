#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-


def bmes_to_span(tags):
    def _to_tree(begin, end, level=0):
        child_begin = begin
        while child_begin < end and level < len(tags[child_begin]):
            if tags[child_begin][level] == 'S':
                yield ((child_begin, child_begin+1), len(tags[child_begin]) - level)
                child_begin += 1
            else:
                while child_begin < end and level < len(tags[child_begin]) and tags[child_begin][level] != 'B':
                    child_begin += 1

                child_end = child_begin + 1
                while child_end < end and level < len(tags[child_end]) and tags[child_end][level] == 'M':
                    child_end += 1

                if child_end < end and tags[child_end][level] == 'E':
                    yield (child_begin, child_end+1), len(tags[child_begin]) - level

                    yield from _to_tree(child_begin, child_end+1, level + 1)

                child_begin = child_end + 1
            print(begin, end, child_begin, level)

    return _to_tree(0, len(tags), 0)


for span, level in bmes_to_span(['BB', 'MM', 'ME', 'ES', 'S']):
    print(span, level)
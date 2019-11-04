#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
from typing import List, Set
from nltk.tree import Tree
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader

corpus = LazyCorpusLoader(
    'ctb8fix', BracketParseCorpusReader, r'.*\.txt')


def to_bmes(length, tag):
    if length == 1:
        return ['S_%s' % tag]
    elif length > 1:
        return ['B_%s' % tag] + ['M_%s' % tag] * (length - 2) + ['E_%s' % tag]
    else:
        raise RuntimeError('length must be big than 0.')


def fix_frag(node: Tree):
    if node.label() != 'FRAG':
        return node

    children = []
    begin = 0
    while begin < len(node):
        if node[begin].label() in {'NN', 'NR'}:
            children.append(Tree('NP', node[begin:begin+1]))
            begin += 1
        elif node[begin].label() == 'NT':
            end = begin+1
            while end < len(node) and node[end].label() == 'NT':
                end += 1
            children.append(Tree('TIME', node[begin:end]))
            begin = end
        elif node[begin].label() in {'VV', 'VC', 'VE'}:
            children.append(Tree('VP', node[begin:begin+1]))
            begin += 1
        else:
            children.append(Tree('O', node[begin:begin+1]))
            begin += 1

    return Tree(node.label(), children)


def flat_unary(node: Tree):
    children = []
    for child in node:
        if isinstance(child, Tree):
            children.append(flat_unary(child))
        else:
            children.append(child)

    if (len(children) == 1 and isinstance(children[0], Tree) and node.label() == children[0].label()):
        return children[0]
    else:
        return Tree(node.label(), children)


def split_NP(node):
    assert node.label().startswith('NP')
    if len(node) >= 3 and \
        not node[0].label().startswith('ADJP') and \
        node[-2].label().startswith('ADJP') and \
        node[-1].label().startswith('NP'):
        begin = len(node) - 2
        while begin > 0 and node[begin].label().startswith('ADJP'):
            begin -= 1
        begin += 1
        node = Tree(node.label(), node[0:begin] + [Tree('NP', node[begin:])])
    elif len(node) == 2 and \
        node[0].label().startswith('DNP') and node[1].label() in {'NN'}:
        node = Tree(node.label(), node[0:1] + [Tree('NP', [node[1]])])
    return node


def split_VP(node):
    assert node.label().startswith('VP')
    if len(node) == 3 and \
        node[0].label() in {'VV'} and \
        node[1].label() == 'AS' and \
        node[-1].label().startswith('NP'):
        node = Tree(node.label(), [Tree('VP', node[0:2]), node[-1]])
    elif len(node) == 2 and \
        node[0].label() in {'VC', 'VV'} and node[1].label().split('-')[0] in {'VP', 'NP'}:
        node= Tree(node.label(), [Tree('VP', node[0:1]), node[1]])
    '''
    elif len(node) > 2 and not node[0].label().startswith('ADVP') and \
        node[-2].label().startswith('ADVP') and \
        node[-1].label().startswith('VP'):
        begin = len(node) - 2
        while begin > 0 and node[begin].label().startswith('ADVP'):
            begin -= 1
        begin += 1
        print(node)
        node = Tree(node.label(), node[0:begin] + [Tree('VP', node[begin:])])
        print(node)
    '''
    return node


def split_CP(node):
    assert node.label().startswith('CP')
    if len(node) >= 2 and node[-1].label() == 'DEC':
        node = Tree(node.label(), node[0:-1] + [Tree('CP', [node[-1]])])

    return node


def flat_NP(label: str, children: List[Tree]):
    assert label.startswith('NP')
    tag_set = set(child.label().split('-')[0] for child in children)
    if len(tag_set.difference({'ADJP', 'NP', 'CC'})) == 0:
        # (ADJP|NP)+
        flat_children = []
        for child in children:
            if isinstance(child, Tree):
                flat_children.extend(child)
            else:
                flat_children.append(child)
        children = flat_children
    elif len(children) == 3 and \
        children[0].label().startswith('NP') and children[1].label().startswith('DNP') and children[2].label().startswith('NP'):
        return children
    elif len(children) == 2 and \
        children[0].label().startswith('DP') and children[1].label().startswith('NP'):
        return children

    return [Tree(label, children)]


def flat_VP(label: str, children: List[Tree]):
    assert label.startswith('VP')
    tag_set = set(child.label().split('-')[0] for child in children)
    if len(tag_set.difference({'VP'})) == 0:

        flat_children = []
        for child in children:
            if isinstance(child, Tree):
                flat_children.extend(child)
            else:
                flat_children.append(child)
        children = flat_children
        return [Tree(label, children)]
    elif len(children) == 3 and \
            children[0].label() in {'VCD', 'VCP', 'VNV', 'VPT', 'VRD', 'VSB'} and \
            children[1].label() == 'AS' and \
            children[-1].label().startswith('NP'):
        # (V*) AS NP
        return [Tree('VP', children[0][0:] + [children[1]]), children[-1]]
    elif len(children) == 2 and \
        children[0].label() in {'VCD', 'VCP', 'VNV', 'VPT', 'VRD', 'VSB'} and \
        children[-1].label().startswith('NP'):
        # (V*) NP
        return [Tree('VP', children[0][0:]), children[-1]]
    elif len(children) == 3 and \
        children[0].label().startswith('VP') and children[1].label() in {'PU', 'CC'} and children[2].label().startswith('VP'):
        return children
    else:
        return [Tree(label, children)]


def flat_LCP(label: str, children: List[Tree]):
    assert label.startswith('LCP')
    if children[-1].label() == 'LC':
        return children[:-1] + [Tree('LCP', children[-1:])]
    else:
        return [Tree(label, children)]


def flat_CLP(label: str, children: List[Tree]):
    assert label.startswith('CLP')
    if len(children) == 2 and children[1].label() == 'M':
        return [Tree(label, [children[0]] + children[1][0:])]
    else:
        return Tree(label, children)


def flat_PP(label: str, children: List[Tree]):
    assert label.startswith('PP')
    assert children[0].label() == 'P'
    if children[0].label() == 'P':
        return [Tree('PP', children[0:1])] + children[1:]
    else:
        return [Tree(label, children)]

def flat_CP(label: str, children: List[Tree]):
    assert label.startswith('CP')
    return children

def flat(node: Tree):

    if node.label().startswith('NP'):
        node = split_NP(node)
    elif node.label().startswith('VP'):
        node = split_VP(node)
    elif node.label().startswith('CP'):
        node = split_CP(node)

    children = []
    for child in node:
        if isinstance(child, Tree):
            children.extend(flat(child))
        else:
            children.append(child)

    # flat phrase
    if node.label().startswith('NP'):
        return flat_NP(node.label(), children)
    if node.label().startswith('VP'):
        return flat_VP(node.label(), children)
    elif node.label().startswith('DNP'):
        assert children[-1].label() == 'DEG'
        children[-1] = Tree('DNP', [children[-1]])
        return children
    elif node.label().startswith('LCP'):
        return flat_LCP(node.label(), children)
    elif node.label().startswith('CLP'):
        return flat_CLP(node.label(), children)
    elif node.label().startswith('PP'):
        return flat_PP(node.label(), children)
    elif node.label().startswith('CP'):
        return flat_CP(node.label(), children)
    else:
        return [Tree(node.label(), children)]


def insert_np(root: Tree):

    def _insert(node):
        assert isinstance(node, Tree)
        if node.label().startswith('NP') and len(node) > 2:
            count = 0
            while count < len(node) and (
                    (node[count].label().startswith('QP') and len(node[count]) > 1) or
                    node[count].label().startswith('DP') or
                    node[count].label().startswith('DNP') or
                    node[count].label().startswith('CP') or
                    node[count].label().startswith('IP')):
                count += 1

            if count > 0 and len(node) - count > 1 \
                    and max(node[count:], key=lambda n: n.height()) < 3 \
                    and node[-1].label().startswith('NP'):
                inserted_node = Tree('NP', node[count:])
                node[:] = node[0:count] + [inserted_node]

        for child in node:
            if isinstance(child, Tree):
                _insert(child)

        return node

    return _insert(root)


def preprocess(root: Tree, chenyu: Set[str]):

    def _process(node):
        if isinstance(node, Tree):
            labels = [l for l in node.label().split('-') if len(l) > 0]
            node.set_label(labels)
            for child in node:
                _process(child)

        return node

    def _merge_chenyu(node: Tree):
        phrase2pos = {'NP': 'NN', 'VP': 'VV', 'QP': 'CD'}
        if len(node.leaves()) == 0:
            print(node, file=sys.stderr)
        text = ''.join(node.leaves())
        if len(node) > 1 and node.label()[0:2] in {'NP', 'VP', 'QP'} and text in chenyu:
            print(node, file=sys.stderr)
            node[:] = [Tree(phrase2pos[node.label()[0:2]], [text])]
            print(node, file=sys.stderr)
        else:
            for child in node:
                if isinstance(child, Tree):
                    _merge_chenyu(child)

    def _fix_bracket(node: Tree):

        new_child = []
        for child in node:
            if isinstance(child, Tree):
                new_child.extend(_fix_bracket(child))
            else:
                new_child.append(child)

        lefts = '“「『【〈[《（'
        rights = '”」』】〉]》）'

        if node.label().startswith('NP'):
            left = ''.join(new_child[0].leaves())
            right = ''.join(new_child[-1].leaves())
            if left in lefts and right in rights:
                if len(node) > 3:
                    return [new_child[0], Tree(node.label(), new_child[1:-1]), new_child[-1]]
                elif len(node) == 3 and isinstance(node[1], Tree) and node[1].label().startswith('NP'):
                    return list(node)

        return [Tree(node.label(), new_child)]

    root = fix_frag(root)
    try:
        root = insert_np(root)

        root = _fix_bracket(root)[0]

        _merge_chenyu(root)
    except Exception as e:
        print(root, file=sys.stderr)

    if root.label() != 'S':
        root = Tree('S', [root])
    return _process(root)


def dict_flat(words, root: Tree):

    def is_word(node):
        if isinstance(node, Tree):
            if 'NP' in node.label():
                if 'TMP' in node.label() or 'PN' in node.label() or 'SHORT' in node.label() and 'APP' not in node.label():
                    return True
                else:
                    text = ''.join(node.leaves())
                    print(text)
                    return text in words
            elif 'QP' in node.label() or 'DP' in node.label():
                return True
        return False

    def _flat(node):
        if len(node) == 1 and not isinstance(node[0], Tree):
            return [node]

        new_children = []
        for child in node:
            if isinstance(child, Tree):
                if 'NONE' in child.label():
                    continue
                new_children.extend(_flat(child))
            else:
                new_children.append(child)

        if is_word(node):
            return [Tree('-'.join(node.label()), new_children)]
        elif 'S' in node.label():
            return Tree('S', new_children)
        else:
            return new_children

    root.set_label('S')
    return _flat(root)


def load_dict(files: List[str], sep='\t'):
    words = set()
    for file in files:
        with open(file) as input:
            for line in tqdm(input, desc='load %s' % file):
                word, *_ = line.strip().split(sep)
                if len(word) > 1:
                    words.add(word)
    print('word size: %s' % len(words), file=sys.stderr)
    # assert '大学生运动会' in words
    return words


pos_tags = [
    'AD', #副词
    'AS', #体标记
    'BA', #把字句标记
    'CC', #coordination conj联合短语的标记“和、或”
    'CD', #数字
    'CS', #subordinating conjunction从属连词“如果，即使”
    'DEC', #关系从句中的“的” 一个星期的访问   修饰性
    'DEG', #associative 定语“的”  上海的工业设施   领属
    'DER', #得
    'DEV', #地
    'DT', #限定词
    'ETC', #等，等等
    'FW', #外来词
    'IJ', #interjection感叹词“嗯”等
    'JJ', #名词做定语
    'LB', #被字句标记
    'LC', #localizer方位词（上下左右内外）
    'M', #度量衡（包括量词）
    'MSP', #一些particles虚词，实际为（来，所，以）
    'NN', #普通名词
    'NR', #专有名词
    'NT', #时间名词
    'OD', #ordinal 序数词
    'ON', #onomatopoeia 拟声词
    'P', #介词
    'PN', #代词
    'PU', #标点
    'SB', #短被动句
    'SP', #句末particle “了”等。
    'VA', #谓词性形容词
    'VC', #copula 系动词是
    'VE', #“有”作为动词
    'VV', #其他动词
    'URL', # url
]


def prune(root: Tree):
    def _prune(node: Tree):
        new_children = []
        for child in node:
            if isinstance(child, Tree):
                label = child.label()
                if 'NONE' in child.label():
                    continue
                if len(label) == 1 and label[0] in pos_tags:
                    new_children.append(Tree(label[0], child))
                else:
                    new_children.extend(_prune(child))
            else:
                new_children.append(child)

        if len(new_children) <= 1:
            return new_children
        elif 'NP' in node.label() or 'QP' in node.label() or 'DP' in node.label() \
                or 'TTL' in node.label() or 'TIME' in node.label() \
                or 'PN' in node.label() or 'S' in node.label():
                # or 'TMP' in node.label() or 'VP' in node.label() \

            return [Tree('-'.join(node.label()), new_children)]
        else:
            return new_children

    return _prune(root)[0]


def tree_to_bmes(root: Tree):

    def _func(node: Tree):
        chars, tags = [], []
        for child in node:
            if isinstance(child, Tree):
                label = child.label()
                if child.height() == 2:
                    chars.extend(child[0])
                    tags.extend(to_bmes(len(child[0]), ''))
                else:
                    _chars, _tags = _func(child)
                    chars.extend(_chars)
                    tags.extend(_tags)

        if node.label() != 'S':
            tags = [prefix + tag for prefix, tag in zip(to_bmes(len(chars), ''), tags)]

        return chars, tags

    return _func(root)



if __name__ == '__main__':
    chenyu = load_dict(sys.argv[1:])
    for id, sent in enumerate(corpus.parsed_sents()):

        fixed = preprocess(sent, chenyu)
        # flat_sent = dict_flat(words, fixed)

        flat_sent = prune(fixed)
        try:
            flat_sent.pprint(margin=100000)
        except:
            print(sent)
            print(flat_sent)

        '''
        print(tree_to_bmes(flat_sent))

        from nltk.draw.tree import draw_trees
        draw_trees(sent, None, flat_sent)

        if id > 100:
            break
            
        '''



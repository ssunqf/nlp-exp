#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys
from nltk import Tree
from typing import List, Tuple
from pyhanlp import SafeJClass
from pyltp import Segmentor, Postagger, NamedEntityRecognizer

Sentence = SafeJClass('com.hankcs.hanlp.corpus.document.sentence.Sentence')
Word = SafeJClass('com.hankcs.hanlp.corpus.document.sentence.word.Word')
CompoundWord = SafeJClass('com.hankcs.hanlp.corpus.document.sentence.word.CompoundWord')


class TokenizeException(Exception):
    pass


def word_span(text: str, words: List[str]) -> List[Tuple[int, int]]:
    offsets = []
    begin = 0
    for word in words:
        try:
            begin = text.index(word[0], begin)
        except Exception as e:
            pass
        offsets.append((begin, begin+len(word)))
        begin += len(word)
    
    return offsets


class Tokenizer:
    def __init__(self):
        self.analyzer = SafeJClass('com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer')()

    def to_word(self, text: str, with_label=False, with_offset=False):
        words = []
        for word in self.analyzer.analyze(text).wordList:
            if isinstance(word, CompoundWord):
                for subword in word.innerList:
                    if with_label:
                        words.append((subword.value, subword.label))
                    else:
                        words.append(subword.value)
            else:
                if with_label:
                    words.append((word.value, word.label))
                else:
                    words.append(word.value)

        return words

    def to_tree(self, text: str):
        root = []
        offset = 0
        for word in self.analyzer.analyze(text).wordList:
            if isinstance(word, CompoundWord):
                compound = []
                for subword in word.innerList:
                    try:
                        offset = text.index(subword.value, offset)
                    except Exception as e:
                        pass
                    compound.append(Tree(subword.label, [(subword.value, (offset, offset+len(subword.value)))]))
                    offset += len(subword.value)
                root.append(Tree(word.label, compound))
            else:
                try:
                    offset = text.index(word.value, offset)
                except Exception as e:
                    pass
                root.append(Tree(word.label, [(word.value, (offset, offset+len(word.value)))]))
                offset += len(word.value)

        return Tree('S', root)


class LTP:
    def __init__(self):
        self.segmentor = Segmentor()
        self.segmentor.load('/Users/sunqf/Downloads/ltp_data_v3.4.0/cws.model')

        self.tagger = Postagger()
        self.tagger.load('/Users/sunqf/Downloads/ltp_data_v3.4.0/pos.model')

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load('/Users/sunqf/Downloads/ltp_data_v3.4.0/ner.model')

    def __del__(self):
        self.recognizer.release()
        self.tagger.release()
        self.segmentor.release()

    def to_word(self, text: str, with_label=False, with_offset=False):
        words = list(self.segmentor.segment(text))

        if with_label:
            tags = self.tagger.postag(words)
            words = list(zip(words, tags))

        return words

    def to_tree(self, text: str):
        '''
        词性标注集

        LTP 使用的是863词性标注集，其各个词性含义如下表。

        Tag	Description	Example	Tag	Description	Example
        a	adjective	美丽
        b	other noun-modifier	大型, 西式
        c	conjunction	和, 虽然
        d	adverb	很
        e	exclamation	哎
        g	morpheme	茨, 甥
        h	prefix	阿, 伪
        i	idiom	百花齐放
        j	abbreviation	公检法
        k	suffix	界, 率
        m	number	一, 第一
        n	general noun	苹果
        nd	direction noun	右侧
        nh	person name	杜甫, 汤姆
        ni	organization name	保险公司
        nl	location noun	城郊
        ns	geographical name	北京
        nt	temporal noun	近日, 明代
        nz	other proper noun	诺贝尔奖
        o	onomatopoeia	哗啦
        p	preposition	在, 把
        q	quantity	个
        r	pronoun	我们
        u	auxiliary	的, 地
        v	verb	跑, 学习
        wp	punctuation	，。！
        ws	foreign words	CPU
        x	non-lexeme	萄, 翱
        z	descriptive words	瑟瑟，匆匆

        LTP NE识别模块的标注结果采用O-S-B-I-E标注形式，其含义为
        标记	含义
        O	这个词不是NE
        S	这个词单独构成一个NE
        B	这个词为一个NE的开始
        I	这个词为一个NE的中间
        E	这个词位一个NE的结尾
        LTP中的NE 模块识别三种NE，分别如下：
        标记	含义
        Nh	人名
        Ni	机构名
        Ns	地名

        :param text:
        :return:
        '''
        words = list(self.segmentor.segment(text))
        pos_tags = list(self.tagger.postag(words))
        ner_tags = list(self.recognizer.recognize(words, pos_tags))

        offsets = word_span(text, words)

        # ltp分词时不考虑存在的空格，被空格隔离的也可能被切成一个词，需要特别处理
        if offsets[-1][1] > len(text):
            raise TokenizeException()

        root = []
        index = 0
        while index < len(words):
            if ner_tags[index].startswith('B-'):
                compound = [Tree(pos_tags[index], [(words[index], offsets[index])])]
                index += 1
                while index < len(words) and ner_tags[index].startswith('I-'):
                    compound.append(Tree(pos_tags[index], [(words[index], offsets[index])]))
                    index += 1

                # 不符合B-I*-E结构的丢弃
                if index >= len(words) or not ner_tags[index].startswith('E-'):
                    root.extend(compound)
                else:
                    compound.append(Tree(pos_tags[index], [(words[index], offsets[index])]))
                    root.append(Tree(ner_tags[index][2:], compound))
                    index += 1
            elif ner_tags[index].startswith('S-'):
                root.append(Tree(ner_tags[index][2:], [Tree(pos_tags[index], [(words[index], offsets[index])])]))
                index += 1
            else:
                root.append(Tree(pos_tags[index], [(words[index], offsets[index])]))
                index += 1

        return Tree('S', root)


hanlp_tokenizer = Tokenizer()

# ltp_tokenizer = LTP()

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        print(ltp_tokenizer.to_word(line))
        print(ltp_tokenizer.to_tree(line).pos())


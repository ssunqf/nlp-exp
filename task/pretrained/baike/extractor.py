#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import random
import numpy as np
import nltk
import itertools
import sys
import re
from tqdm import tqdm
from typing import List, Tuple

from .base import PhraseLabel

from recognizers_text import Culture, ModelResult
from recognizers_number import NumberRecognizer
from recognizers_number_with_unit import NumberWithUnitRecognizer
from recognizers_date_time import DateTimeRecognizer
from recognizers_sequence import SequenceRecognizer

from .flashtext import KeywordProcessor

url = re.compile(
    "((?:(http|https|Http|Https|rtsp|Rtsp)://(?:(?:[a-zA-Z0-9$\\-_.+!*\\'(),;\\?\\&\\=]|(?:\\%[a-fA-F0-9]{2})){1,64}(?:\\:(?:[a-zA-Z0-9\\$\\-\\_\\.\\+\\!\\*\\'\\(\\)\\,\\;\\?\\&\\=]|(?:\\%[a-fA-F0-9]{2})){1,25})?\\@)?)?(?:(((([a-zA-Z0-9][a-zA-Z0-9\\-]*)*[a-zA-Z0-9]\\.)+((aero|arpa|asia|a[cdefgilmnoqrstuwxz])|(biz|b[abdefghijmnorstvwyz])|(cat|com|coop|c[acdfghiklmnoruvxyz])|d[ejkmoz]|(edu|e[cegrstu])|f[ijkmor]|(gov|g[abdefghilmnpqrstuwy])|h[kmnrtu]|(info|int|i[delmnoqrst])|(jobs|j[emop])|k[eghimnprwyz]|l[abcikrstuvy]|(mil|mobi|museum|m[acdeghklmnopqrstuvwxyz])|(name|net|n[acefgilopruz])|(org|om)|(pro|p[aefghklmnrstwy])|qa|r[eosuw]|s[abcdeghijklmnortuvyz]|(tel|travel|t[cdfghjklmnoprtvwz])|u[agksyz]|v[aceginu]|w[fs]|(δοκιμή|испытание|рф|срб|טעסט|آزمایشی|إختبار|الاردن|الجزائر|السعودية|المغرب|امارات|بھارت|تونس|سورية|فلسطين|قطر|مصر|परीक्षा|भारत|ভারত|ਭਾਰਤ|ભારત|இந்தியா|இலங்கை|சிங்கப்பூர்|பரிட்சை|భారత్|ලංකා|ไทย|テスト|中国|中國|台湾|台灣|新加坡|测试|測試|香港|테스트|한국|xn\\-\\-0zwm56d|xn\\-\\-11b5bs3a9aj6g|xn\\-\\-3e0b707e|xn\\-\\-45brj9c|xn\\-\\-80akhbyknj4f|xn\\-\\-90a3ac|xn\\-\\-9t4b11yi5a|xn\\-\\-clchc0ea0b2g2a9gcd|xn\\-\\-deba0ad|xn\\-\\-fiqs8s|xn\\-\\-fiqz9s|xn\\-\\-fpcrj9c3d|xn\\-\\-fzc2c9e2c|xn\\-\\-g6w251d|xn\\-\\-gecrj9c|xn\\-\\-h2brj9c|xn\\-\\-hgbk6aj7f53bba|xn\\-\\-hlcj6aya9esc7a|xn\\-\\-j6w193g|xn\\-\\-jxalpdlp|xn\\-\\-kgbechtv|xn\\-\\-kprw13d|xn\\-\\-kpry57d|xn\\-\\-lgbbat1ad8j|xn\\-\\-mgbaam7a8h|xn\\-\\-mgbayh7gpa|xn\\-\\-mgbbh1a71e|xn\\-\\-mgbc0a9azcg|xn\\-\\-mgberp4a5d4ar|xn\\-\\-o3cw4h|xn\\-\\-ogbpf8fl|xn\\-\\-p1ai|xn\\-\\-pgbs0dh|xn\\-\\-s9brj9c|xn\\-\\-wgbh1c|xn\\-\\-wgbl6a|xn\\-\\-xkc2al3hye2a|xn\\-\\-xkc2dl3a5ee0h|xn\\-\\-yfro4i67o|xn\\-\\-ygbi2ammx|xn\\-\\-zckzah|xxx)|y[et]|z[amw]))|((25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9])\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9]|0)\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[1-9]|0)\\.(25[0-5]|2[0-4][0-9]|[0-1][0-9]{2}|[1-9][0-9]|[0-9]))))(?:\\:\\d{1,5})?)(\\/(?:(?:[a-zA-Z0-9\\;\\/\\?\\:\\@\\&\\=\\#\\~\\-\\.\\+\\!\\*\\'\\(\\)\\,\\_])|(?:\\%[a-fA-F0-9]{2}))*)?")


# http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8
class RulePhraseExtractor:
    unused = {
        'a', 'ad', 'ag', 'al', 'begin', 'end', 'bg', 'bl',
        'c', 'cc', 'd', 'dg', 'dl', 'e', 'end', 'f', 'h', 'i',
        'k', 'l', 'nl', 'o', 'p', 'pba', 'pbei', 's', 'u', 'ud', 'ude1',
        'ude2', 'ued3', 'udeng', 'udh', 'ug', 'uguo', 'uj', 'ul', 'ule',
        'ulian', 'uls', 'usuo', 'uv', 'uyy', 'uz', 'uzhe', 'uzhi', 'v', 'vd', 'vf',
        'vg', 'vi', 'vl', 'vn', 'vshi', 'vx', 'vyou',
        'w', 'wb', 'wd', 'wf', 'wd', 'wt', 'ww', 'wyy', 'wyz',
        'y', 'yg', 'z', 'zg'
    }

    specials = {
        'g', 'gb', 'gbc', 'gc', 'gg', 'gi', 'gm', 'gp', 'j', 'm', 'mg', 'Mg', 'mq',
        'n', 'nb', 'nba', 'nbc', 'nbp', 'nf', 'ng', 'nh', 'nhb', 'nhm',
        'ni', 'nic', 'nis', 'nit', 'nm', 'nmc', 'nn', 'nnd', 'nnt', 'nr', 'nr1', 'nr2', 'nrf', 'nrj',
        'ns', 'nsf', 'nt', 'ntc', 'ntcb', 'ntcf', 'ntch', 'nth', 'nto', 'nts', 'ntu', 'nx', 'nz',
        'q', 'qg', 'qt', 'qv', 'r', 'rg', 'Rg', 'rr', 'ry' 'rys', 'ryt', 'ryv', 'rz', 'rzs',
        't', 'tg', 'wb', 'xu', 'x', 'wh'
    }

    def __init__(self):
        # rule compound word
        self.entity_grammar = """
            NAME1: {<nr.*><nx><n.*>*<nr.*>|<nr.*>}
            NAME2: {<POSITION.*><NAME1>}
                   {<NAME1><nnt|nnd>}
            TIME: {<m><qt>(<m|q.*|t|b>*<qt|t|b>)*}
                  {<t>+}
            NUMEX: {<m><wb|mq>}
                   {<m.*>+<q>?}
            MEASUREX: {<m><a>?<q.*|n_人>}
            ADDRESS: {<url>}
            LOCATION: {<ns.*>+<n>*<f|d|s>}
                      {<ns.*>+<n.*>*<m><q>}
                      {<ns.*>+}

            ORGANIZATION1: {<nt.*>}
                           {<nz|LOCATION.*><nis|v_学会>}
            ORGANIZATION2: {<ORGANIZATION1><vn|m|n.*><ORGANIZATION1>}
                           {<ORGANIZATION1|LOCATION.*><n.*|m.*>*<ni.*|v_学会>}
                           {<LOCATION.*><ORGANIZATION.*>}
                           {<ORGANIZATION1><n><cc><v>*<n><ni.*|v_学会>}<nnt>

            POSITION1: {<nnt|nnd>}
            POSITION2: {<ORGANIZATION.*|LOCATION.*|nz><POSITION1>}

            FOOD: {<LOCATION.*|ORGANIZATION.*|nz|a>?<nf>}
            COMPOUND: {<ns.*|nr|nt.*><n>+<nis|nt.*>}
                      {<ns>*<nt*><nnt>}
                      {<nr|ns><nz>}
                      {<ns.*><n.*|vn|z>*<nnt>}
                      <w-、>{<f><n.*|vn>+}<w-、>
            TITLE: <w_《|w_【>{<.*>+}<w_》|w_】>
                   }<w_》|w_】><.*>*<w_《|w_【>{
        """

        self.noise_grammar = """
            IGNORE: {<a|ad|ag|al|b|c|d.*|e|f|k|l|o|p.*|u.*|vshi|vyou|w.*|y|vi>}   
            IGNORE1: {<ad><v.*>}
                    {<p><n.*>+<v.*>}
                    {<p><n.*>+<f>}
                    {<v.*><.*>+<v.*>}
                    {<n.*>+<ude.*><n.*>+}
                    {<cc><n.*>+<a>}
                    {<d><v.*><ule><a>?<n.*>}
            NOISE: {<.*><IGNORE><.*>}
                   {<.*><IGNORE>}
                   {<IGNORE><.*>}

        """

        self.entity_parser = nltk.RegexpParser(self.entity_grammar)
        self.noise_parser = nltk.RegexpParser(self.noise_grammar)

    def extract(self, words: List[str], tags: List[str], spans: List[Tuple[int, int]]) -> List[PhraseLabel]:
        labels = [PhraseLabel(begin, end, hanlp=True)  # tag in self.specials)
                  for (begin, end), word, tag in zip(spans, words, tags)
                  # if tag in self.specials or tag in self.unused
                  ]

        for (begin, end), word, tag in zip(spans, words, tags):
            if tag in self.specials and end - begin > 1:
                mid = random.randint(begin + 1, end - 1)
                labels.append(PhraseLabel(begin, mid, hanlp=False))
                labels.append(PhraseLabel(mid, end, hanlp=False))

        terms = [(
            (begin, end, word),
            '%s_%s' % (tag, word) if word in ['《', '》', '（', '）', '【', '】', '学会', '人'] else tag)
            for (begin, end), word, tag in zip(spans, words, tags)]

        chunks = self.entity_parser.parse(terms)
        for node in chunks.subtrees(filter=lambda n: n.height() != chunks.height()):
            leaves = node.leaves()
            # print('good', ' '.join([leaf[0][2] for leaf in leaves]))
            if len(leaves) > 1:
                labels.append(PhraseLabel(leaves[0][0][0], leaves[-1][0][1], hanlp=True))

        chunks = self.noise_parser.parse(terms)
        for node in chunks.subtrees(filter=lambda n: n.height() != chunks.height()):
            leaves = node.leaves()
            # print('bad', ' '.join([leaf[0][2] for leaf in leaves]))
            if len(leaves) > 1:
                labels.append(PhraseLabel(leaves[0][0][0], leaves[-1][0][1], hanlp=False))

        return labels


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


'''
class PhraseNode:
    __slots__ = ['begin', 'end', 'children']
    def __init__(self, begin, end, children=None):
        self.begin = begin
        self.end = end
        self.children = children if children else []

class Tree:
    def __init__(self, length):
        self.root = PhraseLabel(0, length)

    def insert(self, begin, end):
        def _insert(node, begin, end):
            if node.begin == begin and node.end == end:
                return node
            if node.begin <= begin < node.end <= node.end:
                if len(node.children) == 0:
                    new_node = PhraseNode(begin, end)
                    node.children.append(new_node)
                    return new_node

                left = 0
                while left < len(node.children) and node.children[left].begin < begin:
                    left += 1
                right = len(node.children) - 1
                while right >= 0 and node.children[right].end > end:
                    right -= 1

                if left == right:

                if left < right:
                    new_node = PhraseNode(begin, end, children=node.children[begin:end])
                    node.children = node.children[:left] + [new_node] + node.children[right:]
                    return new_node
                else:
                    new_node = PhraseNode(begin, end)
                    if left == 0:
                        node.children = [new_node] + node.children
                    elif left >= len(node.children):
                        node.children.append(new_node)
'''


class RuleExtractor:
    def __init__(self):
        number_recognizer = NumberRecognizer(Culture.Chinese)
        self.number_model = number_recognizer.get_number_model()
        self.ordinal_model = number_recognizer.get_ordinal_model()
        self.percentage_model = number_recognizer.get_percentage_model()

        number_with_unit = NumberWithUnitRecognizer(Culture.Chinese)
        self.age_model = number_with_unit.get_age_model()
        self.currency_model = number_with_unit.get_currency_model()
        self.dimension_model = number_with_unit.get_dimension_model()
        self.temperature_model = number_with_unit.get_temperature_model()

        date_time = DateTimeRecognizer(Culture.Chinese)
        self.datetime_model = date_time.get_datetime_model()

        sequence = SequenceRecognizer(Culture.Chinese)
        self.phone_number_model = sequence.get_phone_number_model()
        self.email_model = sequence.get_email_model()
        self.guid_model = sequence.get_guid_model()
        self.hashtag_model = sequence.get_hashtag_model()
        self.ip_address_model = sequence.get_ip_address_model()
        self.mention_model = sequence.get_mention_model()
        self.url_model = sequence.get_url_model()

    def extract(self, text: str, phrases: List[PhraseLabel]):
        rule_phrases = [PhraseLabel(res.start, res.end + 1, **{res.type_name: True, 'rule': True})
                        for model in [self.ordinal_model, self.percentage_model,
                                      self.age_model, self.currency_model, self.dimension_model, self.temperature_model,
                                      self.datetime_model,
                                      self.phone_number_model, self.email_model, self.guid_model, self.hashtag_model,
                                      self.ip_address_model, self.url_model]
                        for res in model.parse(text) if res.end - res.start >= 2
                        ]

        return rule_phrases


class Extractor:
    max_length = 20
    close_pattern = re.compile('(（[^）]+）|【[^】]+】|《[^》]+》)')

    stopwords = {
        "、", "。", "〈", "〉", "《", "》", "一", "一切", "一则", "一方面", "一旦", "一来", "一样", "一般", "七", "万一", "三", "上下", "不仅", "不但",
        "不光", "不单", "不只", "不如", "不怕", "不惟", "不成", "不拘", "不比", "不然", "不特", "不独", "不管", "不论", "不过", "不问", "与", "与其", "与否",
        "与此同时", "且", "两者", "个", "临", "为", "为了", "为什么", "为何", "为着", "乃", "乃至", "么", "之", "之一", "之所以", "之类", "乌乎", "乎",
        "乘", "九", "也", "也好", "也罢", "了", "二", "于", "于是", "于是乎", "云云", "五", "人家", "什么", "什么样", "从", "从而", "他", "他人", "他们",
        "以", "以便", "以免", "以及", "以至", "以至于", "以致", "们", "任", "任何", "任凭", "似的", "但", "但是", "何", "何况", "何处", "何时", "作为",
        "你", "你们", "使得", "例如", "依", "依照", "俺", "俺们", "倘", "倘使", "倘或", "倘然", "倘若", "借", "假使", "假如", "假若", "像", "八", "六",
        "兮", "关于", "其", "其一", "其中", "其二", "其他", "其余", "其它", "其次", "具体地说", "具体说来", "再者", "再说", "冒", "冲", "况且", "几", "几时",
        "凭", "凭借", "则", "别", "别的", "别说", "到", "前后", "前者", "加之", "即", "即令", "即使", "即便", "即或", "即若", "又", "及", "及其", "及至",
        "反之", "反过来", "反过来说", "另", "另一方面", "另外", "只是", "只有", "只要", "只限", "叫", "叮咚", "可", "可以", "可是", "可见", "各", "各个",
        "各位", "各种", "各自", "同", "同时", "向", "向着", "吓", "吗", "否则", "吧", "吧哒", "吱", "呀", "呃", "呕", "呗", "呜", "呜呼", "呢", "呵",
        "呸", "呼哧", "咋", "和", "咚", "咦", "咱", "咱们", "咳", "哇", "哈", "哈哈", "哉", "哎", "哎呀", "哎哟", "哗", "哟", "哦", "哩", "哪",
        "哪个", "哪些", "哪儿", "哪天", "哪年", "哪怕", "哪样", "哪边", "哪里", "哼", "哼唷", "唉", "啊", "啐", "啥", "啦", "啪达", "喂", "喏", "喔唷",
        "嗡嗡", "嗬", "嗯", "嗳", "嘎", "嘎登", "嘘", "嘛", "嘻", "嘿", "四", "因", "因为", "因此", "因而", "固然", "在", "在下", "地", "多", "多少",
        "她", "她们", "如", "如上所述", "如何", "如其", "如果", "如此", "如若", "宁", "宁可", "宁愿", "宁肯", "它", "它们", "对", "对于", "将", "尔后",
        "尚且", "就", "就是", "就是说", "尽", "尽管", "岂但", "己", "并", "并且", "开外", "开始", "归", "当", "当着", "彼", "彼此", "往", "待", "得",
        "怎", "怎么", "怎么办", "怎么样", "怎样", "总之", "总的来看", "总的来说", "总的说来", "总而言之", "恰恰相反", "您", "慢说", "我", "我们", "或", "或是",
        "或者", "所", "所以", "打", "把", "抑或", "拿", "按", "按照", "换句话说", "换言之", "据", "接着", "故", "故此", "旁人", "无宁", "无论", "既",
        "既是", "既然", "时候", "是", "是的", "替", "有", "有些", "有关", "有的", "望", "朝", "朝着", "本", "本着", "来", "来着", "极了", "果然", "果真",
        "某", "某个", "某些", "根据", "正如", "此", "此外", "此间", "毋宁", "每", "每当", "比", "比如", "比方", "沿", "沿着", "漫说", "焉", "然则",
        "然后", "然而", "照", "照着", "甚么", "甚而", "甚至", "用", "由", "由于", "由此可见", "的", "的话", "相对而言", "省得", "着", "着呢", "矣", "离",
        "第", "等", "等等", "管", "紧接着", "纵", "纵令", "纵使", "纵然", "经", "经过", "结果", "给", "继而", "综上所述", "罢了", "者", "而", "而且",
        "而况", "而外", "而已", "而是", "而言", "能", "腾", "自", "自个儿", "自从", "自各儿", "自家", "自己", "自身", "至", "至于", "若", "若是", "若非",
        "莫若", "虽", "虽则", "虽然", "虽说", "被", "要", "要不", "要不是", "要不然", "要么", "要是", "让", "论", "设使", "设若", "该", "诸位", "谁",
        "谁知", "赶", "起", "起见", "趁", "趁着", "越是", "跟", "较", "较之", "边", "过", "还是", "还有", "这", "这个", "这么", "这么些", "这么样",
        "这么点儿", "这些", "这会儿", "这儿", "这就是说", "这时", "这样", "这边", "这里", "进而", "连", "连同", "通过", "遵照", "那", "那个", "那么", "那么些",
        "那么样", "那些", "那会儿", "那儿", "那时", "那样", "那边", "那里", "鄙人", "鉴于", "阿", "除", "除了", "除此之外", "除非", "随", "随着", "零",
        "非但", "非徒", "靠", "顺", "顺着", "首先",
        "︿", "！", "＃", "＄", "％", "＆", "（", "）", "＊", "＋", "，",
        "０", "１", "２", "３", "４", "５", "６", "７", "８", "９",
        "：", "；", "＜", "＞", "？", "＠", "［", "］", "｛", "｜", "｝", "～", "￥"}

    stopwords_max_length = max([len(s) for s in stopwords])

    pairs = {'\[': '\]', '【': '】', '\(': '\)', '（': '）', '“': '”', '{': '}', '｛': '｝', '《': '》', '〈': '〉'}
    unclosed = re.compile('(^(%s)|(%s)$)' % (
        '|'.join(['[^%s]*%s' % (l, r) for l, r in pairs.items()]),
        '|'.join(['%s[^%s]*' % (l, r) for l, r in pairs.items()])
    ))

    def __init__(self, keyword_extractor: KeywordProcessor = None):
        self.distant_extractor = keyword_extractor

    def extract(self, text: str, baike_labels: List[PhraseLabel]):
        # baike_labels.extend(self.rule_extractor.extract(text, baike_labels))
        baike_labels = self._make_negatives(text, baike_labels)
        labeled_offsets = set((label.begin, label.end) for label in baike_labels)

        '''
        tree = segmentor.to_tree(text)
        words = [word for (word, _),  _ in tree.pos()]
        labels = [label for _, label in tree.pos()]
        spans = [span for (_, span), _ in tree.pos()]


        rule_labels = [label for label in self.rule_extractor.extract(words, labels, spans)
                       if (label.begin, label.end) not in labeled_offsets]
        labeled_offsets.update((label.begin, label.end) for label in rule_labels)

        rule_labels.extend(self._word_negatives(spans, 'hanlp'))

        distant_labels = [label for label in self.distant(text, words) if (label.begin, label.end) not in labeled_offsets]

        labeled_offsets.update((label.begin, label.end) for label in distant_labels)
        unlabels = []
        for begin in range(len(text)-2):
            end = random.randint(begin + 2, min(begin+self.max_length, len(text)))
            if (begin, end) not in labeled_offsets:
                unlabels.append(PhraseLabel(begin, end, unlabel=False))
        '''
        return np.array(baike_labels)  # + rule_labels + distant_labels + unlabels)

    '''
    def distant(self, text, words, offsets):

        seg_text = ' '.join(words)
        offset_mapping = {}
        seg_begin = 0
        for (begin, end), word in zip(offsets, words):
            try:
                seg_begin = seg_text.index(word, seg_begin)
                offset_mapping[seg_begin] = begin
                offset_mapping[seg_begin + len(word)] = end
            except Exception as e:
                print(e)

            seg_begin += len(word)

        distant_labels = []
        for keyword, seg_begin, seg_end in self.distant_extractor.extract_keywords(text, span_info=True):
            if seg_begin in offset_mapping and seg_end in offset_mapping:
                begin = offset_mapping[seg_begin]
                end = offset_mapping[seg_end]
                distant_labels.append(PhraseLabel(begin, end, distant=True))

        return distant_labels


    def distant(self, text: str, words: List[str]):
        return [PhraseLabel(begin, end, distant=True)
                for keyword, begin, end in self.distant_extractor.extract_keywords(text, words, span_info=True)]
    '''

    def _make_negatives(self, text: str, labels: List[PhraseLabel]):
        labels = sorted(labels, key=lambda p: (p.begin, p.end))
        negatives = []

        negatives.extend(self._boundary(text, labels, "gold"))
        negatives.extend(self._pairwise(text, labels, "gold"))
        negatives.extend(self._intersect(labels, len(text), "gold"))

        noises = []
        alnum = r'^([A-Za-z][A-Za-z]|\d\d)$'
        noises.extend([p for p in self._internal(labels, len(text), "unlabel")
                          # if text[p.begin:p.end] not in self.distant_extractor
                          # or (p.begin > 0 and re.match(alnum, text[p.begin-1:p.begin+1], re.IGNORECASE))
                          # or (p.end < len(text) and re.match(alnum, text[p.end-1:p.end+1], re.IGNORECASE))
                          ])

        noises.extend(self._external(text, labels, "unlabel"))

        return labels + negatives + noises

    def _pairwise(self, text, phrases: List[PhraseLabel], tag: str):
        first_it, second_it = itertools.tee(phrases)
        next(second_it, None)
        for first, second in zip(first_it, second_it):
            if second.begin - first.end < self.max_length:
                for left in range(first.begin, first.end):
                    for right in range(second.begin + 1, second.end + 1):
                        if left != first.begin or right != second.end:
                            yield PhraseLabel(left, right, **{tag: False})

                if self.unclosed.match(text[first.end:second.begin]):
                    yield PhraseLabel(first.end, second.begin, **{tag: False})
                    yield PhraseLabel(first.begin, second.end, **{tag: False})

    def _internal(self, phrases: List[PhraseLabel], length, tag: str):
        for phrase in phrases:
            for mid in range(phrase.begin + 1, phrase.end):
                if mid - phrase.begin > 1:
                    yield PhraseLabel(phrase.begin, mid, **{tag: False})

                if phrase.end - mid > 1:
                    yield PhraseLabel(mid, phrase.end, **{tag: False})

    def _intersect(self, phrases: List[PhraseLabel], length, tag: str):
        for phrase in phrases:
            for mid in range(phrase.begin + 1, phrase.end):
                for left in range(max(0, phrase.begin - self.max_length), phrase.begin):
                    yield PhraseLabel(left, mid, **{tag: False})

                for right in range(phrase.end + 1, min(phrase.end + self.max_length, length) + 1):
                    yield PhraseLabel(mid, right, **{tag: False})

    def _boundary(self, text: str, phrases: List[PhraseLabel], tag: str):
        for phrase in phrases:
            for length in range(min(self.stopwords_max_length, phrase.begin), 0, -1):
                if text[phrase.begin-length:phrase.begin] in self.stopwords:
                    for left in range(phrase.begin-length, phrase.begin):
                        yield PhraseLabel(left, phrase.end, **{tag: False})
                    break
            for length in range(1, min(self.stopwords_max_length, len(text) - phrase.end) + 1):
                if text[phrase.end:phrase.end + length] in self.stopwords:
                    for right in range(phrase.end + 1, phrase.end + length + 1):
                        yield PhraseLabel(phrase.begin, right, **{tag: False})
                    break

        first_it, second_it = itertools.tee(phrases)
        next(second_it, None)
        for first, second in zip(first_it, second_it):
            if 0 < second.begin - first.end <= self.stopwords_max_length:
                if text[first.end:second.begin] in self.stopwords:
                    yield PhraseLabel(first.begin, second.end, **{tag: False})

    def _external(self, text, phrases: List[PhraseLabel], tag: str):
        length = len(text)
        for phrase in phrases:
            for left in range(max(0, phrase.begin - 5), phrase.begin - 1):
                if text[left:phrase.begin] in self.distant_extractor:
                    for sleft in range(left + 1, phrase.begin):
                        yield PhraseLabel(sleft, phrase.end, **{tag: False})
                    break

            for right in range(min(phrase.end + 5, length), phrase.end + 1, -1):
                if text[phrase.end:right] in self.distant_extractor:
                    for sright in range(phrase.end + 1, right):
                        yield PhraseLabel(phrase.begin, sright, **{tag: False})
                    break


    def _word_negatives(self, phrases: List[Tuple[int, int]], tag: str):
        first_it, second_it, third_it = itertools.tee(phrases, 3)
        next(second_it, None)
        next(third_it, None)
        next(third_it, None)
        for first, second, third in zip(first_it, second_it, third_it):
            if first[1] - first[0] == 1:
                continue
            left = random.randint(first[0], first[1] - 1)
            if second[1] - second[0] == 1:
                yield PhraseLabel(random.randint(first[0] + 1, first[1] - 1), second[1], **{tag: False})
            else:
                right = random.randint(second[0] + 1, second[1])
                if left != first[0] or right != second[1]:
                    yield PhraseLabel(left, right, **{tag: False})

            if third[1] - third[0] == 1:
                yield PhraseLabel(random.randint(first[0] + 1, first[1] - 1), third[1], **{tag: False})
            else:
                right = random.randint(third[0] + 1, third[1])
                if left != first[0] or right != third[1]:
                    yield PhraseLabel(left, right, **{tag: False})

            '''
            for left in range(first[0], first[1]):
                for right in range(second[0]+1, second[1]+1):
                    if left != first[0] or right != second[1]:
                        yield PhraseLabel(left, right, **{tag: False})
                for right in range(third[0]+1, third[1]+1):
                    if left != first[0] or right != third[1]:
                        yield PhraseLabel(left, right, **{tag: False})
            '''

    @classmethod
    def build_from_dict(cls, path: str):
        processor = KeywordProcessor(tokenize_func=lambda p: p)
        processor.from_list(path)

        return cls(processor)


if __name__ == '__main__':
    extractor = Extractor.build_from_dict(sys.argv[1])
    for line in sys.stdin:
        print(line)
        result = extractor.extract(line, [])
        print([(phrase.begin, phrase.end, line[phrase.begin:phrase.end], phrase.labels) for phrase in result])

#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import re

from typing import List, Pattern

hanzi = re.compile(
    u'([^\u0000-\u007f\u00f1\u00e1\u00e9\u00ed\u00f3\u00fa\u00d1\u00c1\u00c9\u00cd\u00d3\u00da\u0410-\u044f\u0406\u0407\u040e\u0456\u0457\u045e])')

phone_number = re.compile(
    r'(?:(\d{3})?[-. ]?(\d{3})[-. ]?(\d{4}))|' # mmm-mmm-mmmm
    r'[1-9]\d{9}|'
    r'(\(\d{3}\)(\d{8}|\d{4}-\d{4}))' # (mmm)mmmmmmmm    (mmmm)mmmm-mmmm
)

integer = re.compile('(?!=\d)[-+]?(([1-9]{1,3}(,\d{3})*)|([1-9]{1,4}(,\d{4})*))')
float_template = r'[-+]?(\d+(\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


numeric = re.compile(float_template)

precent = re.compile(float_template + '(%)')

date = re.compile(
    r'(0?[1-9]|1[012])[-/.](0?[1-9]|[12]\d|3[01])[-/.](1|2)\d\d\d' #mm-dd-yyyy
    r'(0?[1-9]|[12]\d|3[01])[-/.](0?[1-9]|1[012])[-/.](1|2)\d\d\d' #dd-mm-yyyy
    r'(1|2)\d\d\d([-/.])(0?[1-9]|1[012])\2(0[1-9]|[12]\d|3[01])'  #yyyy-mm-dd
)

english = re.compile('[A-Za-z][A-Za-z-.]*')
numeric_english = re.compile('[0-9A-za-z][0-9A-Za-z.&_=\']+')

email = re.compile(
    r'[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?',
    re.IGNORECASE)

url = re.compile(
    r'((?:http|ftp)s?://)?'  # http:// or https://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z]{2,}\.?)|'  # domain...
    r'localhost|'  # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
    r'(?::\d+)?'  # optional port
    r'(?:(/(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9]))?)*\S+)', # path
    re.IGNORECASE)


units = re.compile(
    r'(m/s²|m/s2|°|ʹ|ʹʹ|rad|grad'
    r'|Mm²|Mm2|km²|km2|m²|m2|cm²|cm2|mm²|mm2|µm²|µm2|nm²|nm2|in²|in2|ft²|ft2|yd²|yd2|mi²|mi2|ac|a'
    r'|g/L|mg/dL|ppm|s|m|s|C'
    r'|MAh|kAh|Ah|mAh|C|MA|kA|A|mA|A'
    r'|MV|kV|V|mV'
    r'|MΩ|kΩ|Ω|mΩ'
    r'|kJ|J|kCal|cal'
    r'|THz|GHz|MHz|kHz|Hz|mHz|µHz'
    r'|L/100km|mpg|lx'
    r'|Mm|km|hm|dam|m|dm|cm|mm|µm|nm|pm|in|ft|yd|mi|smi|ly|NM|ftm|fur|ua'
    r'|kg|g|dg|cg|mg|µg|ng|pg|oz|lb|st|t|ton|ct|oz t'
    r'|TW|GW|MW|kW|W|mW|µW|nW|nHz|N/m²'
    r'|GPa|MPa|kPa|hPa|inHg|bar|mbar|mmHg'
    r'|m/s|km/h|mph|K|°C'
    r'|ML|kL|L|dl|cL|mL|km³|km3|m³|m3|dm³|dm3|cm³|cm3|mm³|mm3|in³|in3|ft³|ft3|yd³|yd3|mi³|mi3|af|bsh|tsp|tbsp'
    r'|fl|cup|dpi|kb|mb|gb|tb|AU)', re.IGNORECASE)

numeric_unit = re.compile('%s%s' % (numeric.pattern, units.pattern), re.IGNORECASE)

path = re.compile(
    r'(\w:|[a-z_\-\s0-9.]+)?([/\\]{1,2}[%a-z_\-\s0-9.]+){2,}[\\]?', re.IGNORECASE)

coordinate = re.compile(r'(\d+(\.\d*)?\'(\d+(\.\d*)?")?|(\d+(\.\d*)?"))', re.IGNORECASE)

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
            rstring += chr(inside_code)
        elif (inside_code >= 65281 and inside_code <= 65374 and inside_code != 65292):  # 全角字符（除空格,逗号）根据关系转化
            inside_code -= 65248
            rstring += chr(inside_code)
        else:
            rstring += uchar
    return rstring


def split_hanzi(text):
    return hanzi.sub(r' \1 ', text)


symbols = re.compile('([()&/~\-:*#$+|{}\[\],;<>?!="^]+|\.{2,})')


def replace_entity(text):
    text = split_hanzi(strQ2B(text))
    for word in text.split():
        yield from replace_word(word)


def replace_word(word, split_word=True):
    if hanzi.fullmatch(word) or symbols.fullmatch(word):
        yield ('@zh_char@', word)
    elif email.fullmatch(word):
        yield ('@email@', word)
    elif url.fullmatch(word):
        yield ('@url@', word)
    elif date.fullmatch(word):
        yield ('@date@', word)
    elif english.fullmatch(word):
        yield ('@eng_word@', word)
    elif precent.fullmatch(word):
        yield ('@precent@', word)
    elif integer.fullmatch(word):
        yield ('@integer', word)
    elif numeric.fullmatch(word):
        yield ('@numeric@', word)
    elif numeric_unit.fullmatch(word):
        matched = numeric.match(word)
        m_begin, m_end = matched.span(0)
        yield ('@numeric@', word[m_begin:m_end])
        yield ('@unit@', word[m_end:])
    elif word[-1] in 'NSEW' and coordinate.fullmatch(word[:-1]):
        yield ('@coordinate@', word[:-1])
        yield ('@eng_word@', word[-1])
    elif path.fullmatch(word):
        yield ('@path@', word)
    elif numeric_english.fullmatch(word):
        yield ('@numeric_english@', word)
    elif split_word:
        for sub in symbols.sub(r' \1 ', word).split():
            yield from replace_word(sub, False)
    else:
        yield ('@unk@', word)


def fix_word(word: str):
    for name, pattern in [
        ('@email@', email), ('@url@', url), ('@date@', date),
        ('@eng_word@', english), ('@precent@', precent), ('@integer@', integer),
        ('@numeric@', numeric), ('@unit@', units), ('@coordinate@', coordinate),
        ('@path@', path), ('@numerice_english', numeric_english)]:
        if pattern.fullmatch(word):
            return name

    if len(set(word)) == 1 and len(word) > 5:
        return word[:5]
    return word

class BMESTagger:
    @classmethod
    def tag(cls, token_size: int, type='') -> List[str]:
        if token_size == 1:
            return ['S_' + type]
        elif token_size == 2:
            return ['B_' + type, 'E_' + type]
        else:
            return ['B_' + type] + ['M_' + type] * (token_size - 2) + ['E_' + type]

    @classmethod
    def is_split(cls, tag: str) -> bool:
        return tag.startswith('E_') or tag.startswith('S_')


if __name__ == '__main__':
    print(list(replace_entity('this is a test')))

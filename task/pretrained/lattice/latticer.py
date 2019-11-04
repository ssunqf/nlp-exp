#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
import sys

from recognizers_text import Culture
from recognizers_number import NumberRecognizer
from recognizers_number_with_unit import NumberWithUnitRecognizer
from recognizers_date_time import DateTimeRecognizer
from recognizers_sequence import SequenceRecognizer


if __name__ == '__main__':

    number_recognizer = NumberRecognizer(Culture.Chinese)
    number_model = number_recognizer.get_number_model()
    ordinal_model = number_recognizer.get_ordinal_model()
    percentage_model = number_recognizer.get_percentage_model()

    number_unit_recognizer = NumberWithUnitRecognizer(Culture.Chinese)
    age_model = number_unit_recognizer.get_age_model()
    currency_model = number_unit_recognizer.get_currency_model()
    dimension_model = number_unit_recognizer.get_dimension_model()
    temperature_model = number_unit_recognizer.get_temperature_model()

    datetime_model = DateTimeRecognizer(Culture.Chinese).get_datetime_model()
    sequence_model = SequenceRecognizer(Culture.Chinese).get_phone_number_model()

    recognizers = [number_model, ordinal_model, percentage_model,
                   age_model, currency_model, dimension_model, temperature_model,
                   datetime_model, sequence_model]

    for line in sys.stdin:
        print(line)
        for recognize in recognizers:
            try:
                for res in recognize.parse(line):
                    print(res.__dict__)
            except Exception as e:
                print(e)
        print()

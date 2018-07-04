#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 2017/9/4 19:21
@version: v1.0
@author: FG
@file: RiskIndex.py
@license: CRETEASE Licence
@contact: gangfang6@cretease.cn
@time: 2017/9/4 19:21
'''


class RiskIndex(object):
    def __init__(
            self,
            rank,
            marginal_ratio,
            good_cnt,
            good_precent,
            bad_cnt,
            bad_precent,
            total_cnt,
            total_percent,
            cumulative_odds_ratio,
            cumulative_bad_probability):
        self.rank = rank
        self.marginal_ratio = marginal_ratio
        self.good_cnt = good_cnt
        self.good_precent = good_precent
        self.bad_cnt = bad_cnt
        self.bad_precent = bad_precent
        self.total_cnt = total_cnt
        self.total_percent = total_percent
        self.cumulative_odds_ratio = cumulative_odds_ratio
        self.cumulative_bad_probability = cumulative_bad_probability

    def __repr__(self):
        return 'RiskIndex<%d/%f/%d/%f/%d/%f/%d/%f/%f/%f>' % (self.rank,
                                                          self.marginal_ratio,
                                                          self.good_cnt,
                                                          self.good_precent,
                                                          self.bad_cnt,
                                                          self.bad_precent,
                                                          self.total_cnt,
                                                          self.total_percent,
                                                          self.cumulative_odds_ratio,
                                                          self.cumulative_bad_probability)

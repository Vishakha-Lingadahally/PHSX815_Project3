#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 00:04:22 2021

@author: vishakha
"""
from random import randrange

def distribution (n: int):
    result = []
    for x in range(n):
        sum = randrange(1,7) + randrange(1,7)
        result.append(sum)
    for dice in range(2,13):
        num_of_appearances = result.count(dice)
        percentage = (num_of_appearances / n) * 100
        bar = int(percentage) * '*'
        print("{0:2}:{1:8} ({2:4.1f}%)  {3}".format(dice, num_of_appearances, percentage, bar))
        

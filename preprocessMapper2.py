#!/usr/bin/env python3

import sys

review = []
for line in sys.stdin:
    line = line.strip()
    line = line.split(",")
    value = line[1]
    print(value)

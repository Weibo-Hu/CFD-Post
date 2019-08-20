#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:04:17 2018
    This class is a context manager which show the running time of a code block.
@author: weibo
"""

import time


class timer():

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        # self.end = time.time()
        self.interval = time.time() - self.start
        print("%s took: %0.2fs" % (self.message, self.interval))


# if __name__ == "__main__":
#     with timer("Test 1+1"):
#         1+1

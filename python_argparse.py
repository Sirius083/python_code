# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:03:09 2019

@author: Sirius
"""

import argparse
parser = argparse.ArgumentParser()  # 创建解析对象
# 添加命令行参数和选项
parser.add_argument("--learning_rate", type=float, default=0.01, help="initial learining rate")
parser.add_argument("--max_steps", type=int, default=2000, help="max")
parser.add_argument("--hidden1", type=int, default=100, help="hidden1")
parser.add_argument("--name", default='jlz')
args = parser.parse_args()      # parse_args()从指定的选项中返回一些数据
print(args)

# 在 cmd 输入时，可改变参数的值 
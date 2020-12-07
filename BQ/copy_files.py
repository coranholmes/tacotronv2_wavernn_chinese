#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 14:32
# @Author  : 兮嘉
# @File    : copy_files.py
# @Software: PyCharm


import shutil
import os

fp = os.path.join(os.getcwd(), 'audio_file_list.txt')
with open(fp, 'r') as f:
    for fn in f:
        fn = fn.strip()
        oldname = "/home/jhdxr/Downloads/phone/" + fn
        newname = "/home/jhdxr/Codes/tacotronv2_wavernn_chinese/BQ/wav/" + fn
        print(newname)
        shutil.copyfile(oldname, newname)

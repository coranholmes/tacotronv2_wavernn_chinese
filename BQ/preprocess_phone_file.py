#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/3 13:33
# @Author  : 兮嘉
# @File    : preprocess_phone_file.py
# @Software: PyCharm
# 处理phone_msg


import luadata
import os


lua_path = os.path.join(os.getcwd(), 'data', 'lua_text', 'phone_msg_option_info_data.lua.bytes')
data = luadata.read(lua_path, encoding='utf-8')
data = data['data']
f = open('audio_file_list.txt', 'w')
for dialog_id, dialog in data.items():
    if 4000 <= dialog_id < 5000:
        for sent_id, sent_dict in enumerate(dialog):
            try:
                sentence = sent_dict['sentence'][2:-2]
            except:
                print(sent_dict)
                break
            if '$u' not in sent_dict['renming'] and '$u' not in sentence and '（' not in sentence and len(sentence) > 2:
                filename = 'voice_phone_' + str(dialog_id) + '_' + str(sent_id + 1) + '.wav'
                line = filename + "\t" + sentence + "\n"
                print(line)
                f.writelines(filename+"\n")
                with open(os.path.join(os.getcwd(), 'data', 'audio_txt', filename + ".txt"), 'w') as txt_file:
                    txt_file.writelines(sentence+"\n")
f.close()

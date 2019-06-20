#!/usr/bin/python
# -*- coding: utf-8 -*-
import lxml.html
import requests
import random
import percentage

VISITED_URLS = list()
QUANT_FILES = 0
TOTAL_FILES = 100000

def save_text(text):
    global QUANT_FILES
    filename = '{}'.format(23000+QUANT_FILES)
    with open('corpora/wikipedia/'+filename+'.txt', 'w') as arq:
        arq.write(text)
    percentage.progress(QUANT_FILES, TOTAL_FILES)
    QUANT_FILES = QUANT_FILES + 1

def wrapper(root, url):
    try:
        response = requests.get(root+url)
        response.raise_for_status()
        VISITED_URLS.append(url)
        page = lxml.html.fromstring(response.content)
        
        xpath = '//div[@id="bodyContent"]/descendant::p'
        text = ''.join([p.text_content() for p in page.xpath(xpath)])
        save_text(text)
        
        new_urls = [
            u for u in page.xpath('//a/@href') if u.startswith('/wiki/')
        ]
        return new_urls

    except Exception as err:
        print('Erro: '+str(err))

root = 'https://pt.wikipedia.org/'
urls = ['wiki/Capoeira']

i = 0
while i < len(urls) and QUANT_FILES<TOTAL_FILES:
    new_urls = wrapper(root, urls[i])
    new_urls = [nu for nu in new_urls if nu not in VISITED_URLS]
    urls.extend(new_urls)
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2020

@author: Juan Carlos Gomez

This code split a line of text in a set of features words, emojis/emoticons,
ats, hashtags, links and abreviations, creating a file for each feature.
"""

import re
import emoji
from nltk.tokenize.casual import EMOTICON_RE as emo_re

def clean(line_o, hashs_o, ats_o, links_o):
    """Clean a text (string) by removing from it hashtags, mentions (@) and
    links that are passed in the correspoding lists.

    Keyword arguments:
    line_o -- a string with the text to clean.
    hash_o -- a list with the hashtags to be removed
    ats_o -- a list with the mentions to be removed
    links_o -- a list with the links to be removed

    Output:
    a clean text (string)
    """
    for link in links_o:
        line_o = line_o.replace(link, '')
    for has_h in hashs_o:
        line_o = line_o.replace('#'+has_h, '')
        line_o = line_o.replace('＃'+has_h, '')
    for a_t in ats_o:
        line_o = line_o.replace('@'+a_t, '')
        line_o = line_o.replace('＠'+a_t, '')
    return line_o

def read_abvs(file):
    """Read data from a file line by line, removing the new line chars at the
    end of the line, and store the lines in a list. The lines represent
    abbreviations.

    Keyword arguments:
    file -- string with the file name to read.

    Output:
    a set with the abbreviations of the file as strings.
    """
    abvs_l = []
    with open(file, 'r', encoding='utf-8') as file_r:
        for abv in file_r:
            abvs_l.append(abv.strip())
    return set(abvs_l)

#Definition of configuration parameters
#URL regex pattern
URLS = r"""			# Capture 1: entire matched URL
  (?:
  https?:				# URL protocol and colon
    (?:
      /{1,3}				# 1-3 slashes
      |					#   or
      [a-z0-9%]				# Single letter or digit or '%'
                                       # (Trying not to match e.g. "URI::Escape")
    )
    |					#   or
                                       # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:					# One or more:
    [^\s()<>{}\[\]]+			# Run of non-space, non-()<>{}[]
    |					#   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
  )+
  (?:					# End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)				# balanced parens, non-recursive: (...)
    |					#   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]	# not a space or one of these punct chars
  )
  |					# OR, the following to match naked domains:
  (?:
  	(?<!@)			        # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)			        # not succeeded by a @,
                            # avoid matching "foo.na" in "foo.na@example.com"
  )
"""

W_D = 'D:/Documentos/codeanddata/datasets/2019_twitter_education_degree_v2/'
TWEET_FILE = W_D + 'tweets.txt'
ABVS_LIST_FILE = W_D + 'abreviaturas.txt'
WORDS_FILE = W_D + 'split/words.txt'
EMOJI_FILE = W_D + 'split/emoticons.txt'
HASH_FILE = W_D + 'split/hashtags.txt'
AT_FILE = W_D + 'split/ats.txt'
LINK_FILE = W_D + 'split/links.txt'
ABV_O_FILE = W_D + 'split/abvs.txt'

abv_s = read_abvs(ABVS_LIST_FILE)

i = 0
url_re = re.compile(URLS, re.VERBOSE | re.I | re.UNICODE)
hashtag_re = re.compile('(?:^|\s)[＃#]{1}(\w+)', re.UNICODE)
mention_re = re.compile('(?:^|\s)[＠@]{1}(\w+)', re.UNICODE)

with open(TWEET_FILE, 'r', encoding='utf-8') as tweet_r, open(WORDS_FILE, 'w', encoding='utf-8') as words_w, open(EMOJI_FILE, 'w', encoding='utf-8') as emo_w, open(HASH_FILE, 'w', encoding='utf-8') as hash_w, open(AT_FILE, 'w', encoding='utf-8') as at_w, open(LINK_FILE, 'w', encoding='utf-8') as link_w, open(ABVS_LIST_FILE, 'w', encoding='utf-8') as abv_r:
    for line in tweet_r:
        line = line.strip().lower()
        hashs = hashtag_re.findall(line)
        ats = mention_re.findall(line)
        links = url_re.findall(line)
        line = clean(line, hashs, ats, links)
        emoticons = emo_re.findall(line)
        emojis = [w for w in line if w in emoji.UNICODE_EMOJI]
        words = re.findall('[a-záéíóúñàèìòù][a-záéíóúñàèìòù_-]+', line)
        abvs = [w for w in re.findall('[a-z0-9ñáéíóú+/]+', line) if w in abv_s
                and len(w) > 1]

        words_w.write(' '.join(w for w in words)+'\n')
        abv_r.write(' '.join(w for w in abvs)+'\n')
        emo_w.write(' '.join(w for w in emoticons+emojis)+'\n')
        hash_w.write(' '.join(w for w in hashs)+'\n')
        at_w.write(' '.join(w for w in ats)+'\n')
        link_w.write(' '.join(w for w in links)+'\n')
        i += 1
        print(i, 'tweets')
        
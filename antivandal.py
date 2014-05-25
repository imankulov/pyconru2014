#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of helper functions to work with PAN-WVC-10[1] and PAN-WVC-11[2] datasets.

In order to make these functions work you have to have in your current working
directory downloaded and unzipped archives with original data, in directories
pan-wikipedia-vandalism-corpus-2010 and pan-wikipedia-vandalism-corpus-2011
correspondingly.

Here we use MongoDB as an intermediate storage and environment variables
to configure utils settings.

MongoDB document schema
-----------------------

- ds: dataset source 2010 or 2011
- editid: the id of the edit within the dataset
- editor: editor Wikipedia username or IP address
- oldrevisionid: old revision id
- newrevisionid: new revision id
- diffurl: diff url
- edittime: edit time as a timestamp
- editcomment: the comment of the editor
- articleid: id of the article
- articletitle: title of the article
- vandalism: true / false
- oldrevision: old revision content
- newrevision: new revision content


[1]: http://www.uni-weimar.de/en/media/chairs/webis/research/corpora/corpus-pan-wvc-10/
[2]: http://www.uni-weimar.de/en/media/chairs/webis/research/corpora/corpus-pan-wvc-11/
"""
from __future__ import division
import re
import os
import nltk
import time
import random
import pymongo
import requests
import datetime
import string
import lzw
import urllib
from tqdm import tqdm
from csv import DictReader, DictWriter

# env variables
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/vandal')
MONGODB_CORPUS_COLLECTION = os.getenv('MONGODB_CORPUS_COLLECTION', 'corpus')
MONGODB_HTTPCACHE_COLLECTION = os.getenv('MONGODB_HTTPCACHE_COLLECTION', 'httpcache')
MONGODB_USERS_COLLECTION = os.getenv('MONGODB_USERS_COLLECTION', 'users')

# create mongodb connection, database and collection
conn = pymongo.MongoClient(MONGODB_URL)
db = conn.get_default_database()
corpus = db[MONGODB_CORPUS_COLLECTION]
httpcache = db[MONGODB_HTTPCACHE_COLLECTION]
users = db[MONGODB_USERS_COLLECTION]

# misc constants
re_ip = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
re_http_url = re.compile(r'\bhttps?://[\w/%?#&_\-=.,]+[\w/%?#&_\-](?=[\'<>.,]?)')
magic = 'lskmqs'

def ensure_index():
    """
    Ensure MongoDB database has indexes
    """
    corpus.ensure_index([('ds', pymongo.ASCENDING), ('editid', pymongo.ASCENDING)],
                        unique=True, name='unique_id')
    corpus.ensure_index([('vanalism', pymongo.ASCENDING)], name='vandalism')
    httpcache.ensure_index([('key', pymongo.ASCENDING)], name='cache_key')
    users.ensure_index([('name', pymongo.ASCENDING)], name='name')


def import_corpus_2010():
    """
    Import 2010 data to MongoDB
    """
    edits = list(DictReader(open('pan-wikipedia-vandalism-corpus-2010/edits.csv')))
    gold_annotations = list(DictReader(open('pan-wikipedia-vandalism-corpus-2010/gold-annotations.csv')))
    vandalism_ids = {int(a['editid']) for a in gold_annotations if a['class'] == 'vandalism'}
    revision_filenames = get_revision_filenames_2010()
    for edit in tqdm(edits):
        edit = to_int(edit, 'articleid', 'editid', 'newrevisionid', 'oldrevisionid')
        edit = to_timestamp(edit, 'edittime')
        edit.update(ds=2010, vandalism=edit['editid'] in vandalism_ids)
        edit['oldrevision'] = open(revision_filenames[edit['oldrevisionid']]).read()
        edit['newrevision'] = open(revision_filenames[edit['newrevisionid']]).read()
        spec = dict(ds=2010, editid=edit['editid'])
        corpus.update(spec, edit, upsert=True)


def import_corpus_2011():
    """
    Import 2011 data to MongoDB
    """
    edits = list(DictReader(open('pan-wikipedia-vandalism-corpus-2011/edits-en.csv')))
    revision_filenames = get_revision_filenames_2011()
    for edit in tqdm(edits):
        del edit['annotators']
        del edit['totalannotators']
        edit_class = edit.pop('class')
        edit['vandalism'] = edit_class == 'vandalism'
        edit['ds'] = 2011
        edit = to_int(edit, 'articleid', 'editid', 'newrevisionid', 'oldrevisionid')
        edit = to_timestamp(edit, 'edittime')
        edit['oldrevision'] = open(revision_filenames[edit['oldrevisionid']]).read()
        edit['newrevision'] = open(revision_filenames[edit['newrevisionid']]).read()
        spec = dict(ds=2011, editid=edit['editid'])
        corpus.update(spec, edit, upsert=True)


def get_revision_filenames_2010():
    return get_revision_filenames('pan-wikipedia-vandalism-corpus-2010/article-revisions')


def get_revision_filenames_2011():
    return get_revision_filenames('pan-wikipedia-vandalism-corpus-2011/article-revisions-en')


def get_revision_filenames(topdir):
    """
    Helper function returning a dict: {revision_id: path/to/file.txt}
    """
    result = {}
    for dirname, dirs, files in os.walk(topdir):
        if dirname == topdir:
            continue
        for filename in files:
            revid = int(filename.split('.')[0])
            result[revid] = os.path.join(dirname, filename)
    return result


#--- Dumper


def export_collection(filename, collection=corpus, include=None, exclude=None, converters=None, default_values=None):
    """
    Export corpus to a csv file

    :param exclude: list of fields to exclude. By default we exclude
                    "oldrevision" and "newrevision" records
    """
    converters = converters or {}
    default_values = default_values or {}
    all_fields = set(collection.find_one().keys())

    if include is not None:
        # handle include parameter
        returned_fields = include
        fields_dict = {f: False for f in all_fields if f not in include}

    else:
        # handle exclude parameter
        if exclude is None:
            exclude = ['oldrevision', 'newrevision', 'diff_word', 'neg_diff_word']
        exclude = set(exclude)
        exclude.add('_id')

        returned_fields = sorted(all_fields - exclude)
        fields_dict = {f: False for f in exclude}

    with open(filename, mode='w') as fd:
        writer = DictWriter(fd, returned_fields)
        writer.writeheader()
        for record in tqdm(collection.find(fields=fields_dict), total=collection.count()):
            out = {}
            for k in returned_fields:
                v = record.get(k, default_values.get(k))
                if k in converters:
                    v = converters[k](v)
                if isinstance(v, unicode):
                    v = v.encode('utf8')
                if isinstance(v, bool):
                    v = int(v)
                out[k] = v
            writer.writerow(out)


#--- Dataset extension functions

def apply_all():
    apply(cleanup, extend_with_diff, extend_with_basic_text_metrics, extend_with_ratio_metrics)

def apply(*functions):
    """
    A function which applies the sequence of functions `func` to every item of
    the MongoDB-stored dataset. If function returns a dict, it's merged back
    with the item and we update the record
    """
    for record in tqdm(corpus.find(), total=corpus.count()):

        orig_record = record.copy()

        for func in functions:
            ret = func(**record)
            if ret:
                record = dict(record, **ret)

        if record != orig_record:
            corpus.update({'_id': record['_id']}, record)


def cleanup(**record):
    for key in ['difflen' 'commentlen' 'empty_comment' 'sz_ratio' 'ul_ratio'
            'u_ratio' 'd_ratio' 'non_alnum_ratio' 'compressibility'
            'longest_word' 'longest_seq' 'diff_word' 'neg_ul_ratio'
            'neg_u_ratio' 'neg_d_ratio' 'neg_non_alnum_ratio'
            'neg_compressibility' 'neg_longest_word' 'neg_longest_seq'
            'neg_diff_word']:
        record.pop(key, None)
    if record['editcomment'] == 'null':
        return {'editcomment': ''}


def extend_with_diff(oldrevision, newrevision, **record):

    # preprocess the text to replace url with word-alike sequences
    old_rev, old_pr_map = keep_urls_preprocess(oldrevision)
    new_rev, new_pr_map = keep_urls_preprocess(newrevision)

    # split the text to chunks and replace back word-alike sequences with urls
    old_rev_chunks = keep_urls_postprocess(nltk.word_tokenize(old_rev), old_pr_map)
    new_rev_chunks = keep_urls_postprocess(nltk.word_tokenize(new_rev), new_pr_map)

    # make sure all the chunks are unique
    old_rev_set = set(old_rev_chunks)
    new_rev_set = set(new_rev_chunks)

    # find the "positive diff": all the records which have been added by the edit
    diff_set = sorted(new_rev_set.difference(old_rev_set))
    diff_word = u' '.join(diff_set)

    # find the "negative diff": all the record which have been removed by the edit
    neg_diff_set = sorted(old_rev_set.difference(new_rev_set))
    neg_diff_word = u' '.join(neg_diff_set)

    # keep track of old and new urls
    old_rev_urls = set(old_pr_map.values())
    new_rev_urls = set(new_pr_map.values())

    # find the positive diff for URLs
    url_diff_set = sorted(new_rev_urls.difference(old_rev_urls))
    url_diff_word = u' '.join(url_diff_set)
    urls_added = bool(url_diff_set)

    # find the negative diff for URLs
    neg_url_diff_set = sorted(old_rev_urls.difference(new_rev_urls))
    neg_url_diff_word = u' '.join(neg_url_diff_set)
    urls_removed = bool(neg_url_diff_set)

    ret =  {'diff': diff_word,
            'neg_diff': neg_diff_word,
            'urls': url_diff_word,
            'neg_urls': neg_url_diff_word,
            'urls_added': urls_added,
            'urls_removed': urls_removed}

    return ret


def extend_with_basic_text_metrics(oldrevision, newrevision, editcomment, **kw):
    return {
        'difflen': len(newrevision) - len(oldrevision),
        'commentlen': len(editcomment),
        'empty_comment': len(editcomment) == 0,
        'blanking': len(newrevision) == 0,
        'sz_ratio': (len(newrevision) + 1) / (len(oldrevision) + 1),
    }


def extend_with_ratio_metrics(**record):
    ret = {}

    for key in ['diff', 'neg_diff', 'editcomment']:
        value = record[key]

        total_len = len(value)
        upper_len = sum(c in uppercase for c in value)
        lower_len = sum(c in uppercase for c in value)
        digits_len = sum(c in digits for c in value)
        alnum_len = sum(c in alphanum for c in value)

        if value == 0:
            compressed_len = 0
        else:
            compressed_len = len(list(lzw.compress(value.encode('utf8'))))

        ret.update({
            key + '_ul_ratio': None if lower_len == 0 else upper_len / lower_len,
            key + '_u_ratio': None if total_len == 0 else upper_len / total_len,
            key + '_d_ratio': None if total_len == 0 else digits_len / total_len,
            key + '_non_alnum_ratio': None if total_len == 0 else (total_len - alnum_len) / total_len,
            key + '_compressibility': None if compressed_len == 0 else total_len / compressed_len,
        })

    return ret



uppercase = set(string.uppercase)
lowercase = set(string.lowercase)
digits = set(string.digits)
alphanum = set(string.ascii_letters + string.digits)


def find_longest_seq(string):
    prev = None
    sz = 0
    max_sz = 0
    for c in string:
        if c == prev:
            sz += 1
        else:
            prev = c
            max_sz = max(sz, max_sz)
            sz = 1
    max_sz = max(sz, max_sz)
    return max_sz


#--- Wikipedia API helpers

def api_request(action='query', **kw):
    headers = {'User-Agent': 'AntiVandal Wikipedia API client (https://github.com/imankulov/pyconru2014)'}
    kw.update(format='json')
    base_url = 'http://en.wikipedia.org/w/api.php'

    # convert lists to mediawiki standart representation ("|"-separated string)
    for k, v in kw.items():
        if isinstance(v, list):
            kw[k] = '|'.join(v)

    # GET parameters for HTTP request
    params = dict(action=action, **kw)

    # cache hit
    cache_key = urllib.urlencode(sorted(params.iteritems()))
    cache_result = httpcache.find_one({'key': cache_key})
    if cache_result:
        return cache_result['val']

    # cache miss
    resp = requests.get(base_url, headers=headers, params=params)
    json_resp = resp.json()
    httpcache.insert({'key': cache_key, 'val': json_resp})
    return json_resp


def get_page_revisions(page_id, revision_id):
    """
    Get all page revisions, starting 1 Jan 2009 up to revision in question, in
    reverse order (newer revisions first)
    """
    ret = []
    page_id = str(page_id)
    rvend = datetime.datetime(2009, 1, 1).strftime('%s')
    rvprop = 'ids|flags|timestamp|user|size'
    rvlimit = 500
    rvcontinue = None
    while True:
        kwargs = dict(prop='revisions', pageids=page_id, rvstartid=revision_id,
                      rvend=rvend, rvprop=rvprop, rvlimit=rvlimit)
        if rvcontinue is not None:
            kwargs['rvcontinue'] = rvcontinue
        resp = api_request(**kwargs)
        ret += resp['query']['pages'][page_id]['revisions']
        if 'query-continue' not in resp:
            break
        rvcontinue = resp['query-continue']['revisions']['rvcontinue']
    return ret


def fill_editors_info():
    usprop = 'blockinfo|groups|editcount|registration|emailable|gender'

    cursor = corpus.find({}, fields=['editor'])
    seen = set()

    current = set()
    for record in tqdm(cursor, total=corpus.count()):
        editor = record['editor']
        if re_ip.match(editor) or editor in seen:
            continue
        current.add(editor)
        if len(current) >= 50:
            seen.update(current)
            result = api_request(list='users', ususers=current, usprop=usprop)
            users.insert(result['query']['users'])
            current = set()
    if current:
        result = api_request(list='users', ususers=current, usprop=usprop)
        users.insert(result['query']['users'])


#--- Converters

def to_int(obj, *keys):
    for key in keys:
        obj[key] = int(obj[key])
    return obj


def to_timestamp(obj, *keys):
    for key in keys:
        obj[key] = int(time.mktime(time.strptime(obj[key], '%Y-%m-%dT%H:%M:%SZ')))
    return obj


#--- Utils

def keep_urls_preprocess(text):
    preprocessor_map = {}
    rev_preprocessor_map = {}
    def _preprocessor(match):
        url = match.group(0)
        if url in rev_preprocessor_map:
            repl = rev_preprocessor_map[url]
        else:
            repl = magic + ''.join(random.sample(string.ascii_lowercase, 1)[0] for _ in xrange(16))
            preprocessor_map[repl] = url
            rev_preprocessor_map[url] = repl
        return repl
    return re_http_url.sub(_preprocessor, text), preprocessor_map


def keep_urls_postprocess(chunks, preprocessor_map):
    #ret = []
    #for chunk in chunks:
    #    if chunk in preprocessor_map:
    #        chunk = preprocessor_map[chunk]
    #    ret.append(chunk)

    # Slow but rubust algorithm
    ret = []
    for chunk in ret:
        for key, value in preprocessor_map.iteritems():
            chunk = chunk.replace(key, value)
        ret.append(chunk)
    return ret

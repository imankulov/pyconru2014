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
import os
import time
import pymongo
import requests
import datetime
from tqdm import tqdm
from csv import DictReader, DictWriter

# env variables
MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/vandal')
MONGODB_CORPUS_COLLECTION = os.getenv('MONGODB_CORPUS_COLLECTION', 'corpus')


# create mongodb connection, database and collection
conn = pymongo.MongoClient(MONGODB_URL)
db = conn.get_default_database()
corpus = db[MONGODB_CORPUS_COLLECTION]


def ensure_index():
    """
    Ensure MongoDB database has indexes
    """
    corpus.ensure_index([('ds', pymongo.ASCENDING), ('editid', pymongo.ASCENDING)],
                        unique=True, name='unique_id')


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


def export_corpus(filename, exclude=None):
    """
    Export corpus to a csv file

    :param exclude: list of fields to exclude. By default we exclude
                    "oldrevision" and "newrevision" records
    """
    if exclude is None:
        exclude = ['oldrevision', 'newrevision']

    exclude = set(exclude)
    exclude.add('_id')

    all_fields = set(corpus.find_one().keys())
    returned_fields = sorted(all_fields - exclude)
    fields_dict = {f: False for f in exclude}

    with open(filename, mode='w') as fd:
        writer = DictWriter(fd, returned_fields)
        writer.writeheader()
        for record in tqdm(corpus.find(fields=fields_dict), total=corpus.count()):
            record = {k: v.encode('utf8') if isinstance(v, unicode) else v for k, v in record.iteritems()}
            writer.writerow(record)


#--- Dataset extension functions

def apply(func):
    """
    A function which applies the function `func` to every item of the
    MongoDB-stored dataset. If function returns a dict, it's merged back
    with the item and we update the record
    """
    for record in tqdm(corpus.find(), total=corpus.count()):
        ret = func(record)
        if ret:
            new_record = dict(record, **ret)
            if new_record != record:
                corpus.update({'_id': record['_id']}, new_record)


def extend_with_text_metrics(record):
    """
    An `apply` argument. Generates a set of simplest text metrics for the record
    """
    old_rev = record['oldrevision']
    new_rev = record['newrevision']
    return {
        'difflen': len(new_rev) - len(old_rev),
        'commentlen': len(record['editcomment']),
    }


#--- Wikipedia API helpers

def api_request(action='query', **kw):
    headers = {'User-Agent': 'AntiVandal Wikipedia API client (https://github.com/imankulov/pyconru2014)'}
    kw.update(format='json')
    base_url = 'http://en.wikipedia.org/w/api.php'
    resp = requests.get(base_url, headers=headers, params=dict(action=action, **kw))
    return resp.json()


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


#--- Converters

def to_int(obj, *keys):
    for key in keys:
        obj[key] = int(obj[key])
    return obj


def to_timestamp(obj, *keys):
    for key in keys:
        obj[key] = int(time.mktime(time.strptime(obj[key], '%Y-%m-%dT%H:%M:%SZ')))
    return obj

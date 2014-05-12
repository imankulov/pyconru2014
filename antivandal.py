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
from tqdm import tqdm
from csv import DictReader

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
    result = {}
    for dirname, dirs, files in os.walk(topdir):
        if dirname == topdir:
            continue
        for filename in files:
            revid = int(filename.split('.')[0])
            result[revid] = os.path.join(dirname, filename)
    return result


def to_int(obj, *keys):
    for key in keys:
        obj[key] = int(obj[key])
    return obj


def to_timestamp(obj, *keys):
    for key in keys:
        obj[key] = int(time.mktime(time.strptime(obj[key], '%Y-%m-%dT%H:%M:%SZ')))
    return obj

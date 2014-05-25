import time
import numpy as np
import pandas as pd
from antivandal import export_collection, users


base_fields = [
    u'editid',

    u'articleid',
    u'articletitle',
    u'diffurl',
    u'editcomment',
    u'editor',
    u'edittime',
    u'newrevisionid',
    u'oldrevisionid',
    u'vandalism',
]


meta_fields = [
    u'editid',

    u'blanking',
    u'diff',
    u'diff_compressibility',
    u'diff_d_ratio',
    u'diff_non_alnum_ratio',
    u'diff_u_ratio',
    u'diff_ul_ratio',
    u'urls',
    u'urls_added',
    u'editcomment',
    u'editcomment_compressibility',
    u'editcomment_d_ratio',
    u'editcomment_non_alnum_ratio',
    u'editcomment_u_ratio',
    u'editcomment_ul_ratio',
    u'neg_diff',
    u'neg_diff_compressibility',
    u'neg_diff_d_ratio',
    u'neg_diff_non_alnum_ratio',
    u'neg_diff_u_ratio',
    u'neg_diff_ul_ratio',
    u'neg_urls',
    u'urls_removed',
]


def groups_converter(v):
    try:
        v.remove('*')
    except ValueError:
        pass
    return ' '.join(v)


def timestamp_converter(v):
    if v:
        return int(time.mktime(time.strptime(v, '%Y-%m-%dT%H:%M:%SZ')))


def export():
    export_collection('dataset.csv', include=base_fields)
    export_collection('metainfo.csv', include=meta_fields)
    export_collection('users.csv',
                      collection=users,
                      include=['editcount', 'gender', 'groups', 'name', 'registration'],
                      converters={
                          'groups': groups_converter,
                          'registration': timestamp_converter
                      },
                      default_values={
                          'editcount': 0,
                          'gender': 'unknown',
                          'groups': [],
                          'registration': None,
                      }
    )


def split_dataset():
    dataset = pd.read_csv('dataset.csv').sort('editid')
    idx_train, idx_test = split_idx(dataset.shape[0])

    # we don't want put "vandalism" result to the test dataset
    dataset_test_columns = list(dataset.columns)
    dataset_test_columns.remove('vandalism')

    # export train dataset as is
    dataset.iloc[idx_train].to_csv('train.csv', index=False)

    # export test dataset
    dataset.iloc[idx_test].to_csv('test.csv', index=False, cols=dataset_test_columns)

    # export expected test results
    dataset.iloc[idx_test].to_csv('solution.csv', index=False, cols=['editid', 'vandalism'])



def split_idx(dataset_length, p=0.5, seed=1234):
    idx = np.arange(dataset_length)
    np.random.seed(seed)
    np.random.shuffle(idx)
    t = int(dataset_length * p)
    return idx[:t], idx[t:]


if __name__ == '__main__':
    export()

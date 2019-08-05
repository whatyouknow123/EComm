# encoding=utf-8
import datetime
import os
import sys
import itertools
import fire
import numpy as np

os.environ['PYTHON_EGG_CACHE'] = '/tmp/.python-eggs'

reload(sys)
sys.setdefaultencoding('utf8')
from pyspark.sql import SparkSession

s3_dir = 's3://mx-machine-learning/dongxinzhou'

# reference (my blog): http://xtf615.com/2018/05/03/recommender-system-survey/
# the implementation summary:
# 1. itemID, uid_list: (i1, [u1,u2,...uk...,un]) ...
# 2. uid-uid pair, ((u1, u2), [i1]) ...
# 3. uid-uid, itemID_list, ((u1, u2), [i1,i2,...]))
# 4. itemID-itemID, contribution of uid-uid pair to this itemID-itemID, ( (i1,i2), 1/(alpha+len([i1,i2,...])) ), ...
# 5. itemID-itemID, total contribution (i.e., sim) of all uid-uid pair to this itemID-itemID, ( (i1,i2), sim )


# two main args
# average clicked by 275 users for each item in 2019.03.10.
# for each item with its uid_list, sample 2000 users at most to construct user-user pairs
# this is the main reason why swing consume memory a lot !! the larger this value is, the higher the precision is
bin_batch_num = 2000

alpha = 10.0  # the smoothing value in denominator of swing formula


def read_item_uids_click_online_rdd(spark_session, date):
    # Violent modification
    file_dir = '%s/log/%s' % (s3_dir, date)
    online_rdd = spark_session.sparkContext.textFile(file_dir)
    ret_rdd = online_rdd.map(lambda l: l.split(' ')).map(lambda r: (r[0], r[1:]))
    return ret_rdd


def parse_user_pair(row):
    '''
    :param row: i1, [u1, u2, ..., un]  user_list who have clicked the item
    :return: [((u1,u2), [i1]), ..., ...]  i1 is commonly clicked by u1 and u2
    '''
    # the number of users who click this key item is usually very large, so sample some users to construct user_pair
    sorted_list = sorted(row[1])  # sort to ensure that the key (a,b) and (b,a) can be reduced together
    comb_user_pairs = list(itertools.combinations(sorted_list, 2))
    return [(pair, [row[0]]) for pair in comb_user_pairs]


def split_item_users(row):
    itemID = row[0]
    uids = row[1]
    np.random.shuffle(uids)

    n_bins = int((1.0 * len(uids) / bin_batch_num))
    all_item_uids = []
    # split bins to decrease time complexity
    for i in range(n_bins):
        per_bin_uids = sorted(uids[i*bin_batch_num: (i+1)*bin_batch_num])  # sort !!
        all_item_uids.append((itemID, per_bin_uids))

    # remaining
    remain_uids = sorted(uids[n_bins*bin_batch_num:])  # sort !
    all_item_uids.append((itemID, remain_uids))

    return all_item_uids


def parse_item_pair(row):
    '''
    :param row: (u1,u2), [i1,i2,...,in], item list that have been clicked by both u1 and u2
    :return: [ ((i1,i2), contribution value by pair u1,u2), ... , ]
    '''
    # the number of commonly clicked items by the key user-pair, which is usually not large, so directly combinations
    user_pair_common_click_item_num = len(row[1])
    sorted_list = sorted(row[1])  # sort to ensure that the key (a,b) and (b,a) can be reduce together
    comb_item_pairs = list(itertools.combinations(sorted_list, 2))  # (i1, i2)
    # the contribution by u1-u2 pair for each item-pair
    contribution_value_by_user_pair = 1.0 / (alpha + user_pair_common_click_item_num)  # alpha is the smoothing param
    # [(i1, i2), contribute_value, ..., ]
    return [(item_pair, contribution_value_by_user_pair) for item_pair in comb_item_pairs]


def deterministic_random_partition(item_uids_click_rdd, partition_num):
    def random_key_func(row):
        idx = np.random.randint(0, 1000000)
        return idx, (row[0], row[1])  # k-v format important !!!
    item_uids_click_rdd = item_uids_click_rdd.map(random_key_func)
    item_uids_click_rdd = item_uids_click_rdd.partitionBy(partition_num, lambda idx: idx) # idx is the first element of each line
    item_uids_click_rdd = item_uids_click_rdd.map(lambda (idx, v): v)  # delete idx
    return item_uids_click_rdd


def run(spark_session, date):
    date = str(date)
    # # online df
    online_item_uids_click_rdd = read_item_uids_click_online_rdd(spark_session, date)
    online_item_uids_click_rdd = online_item_uids_click_rdd.repartition(1000)
    # reduce first
    online_item_uids_click_rdd = online_item_uids_click_rdd.reduceByKey(
                                    lambda v1, v2: list(set(v1 + v2)))

    # split uid sequences to many sub-sequences !!! ensure for each (item, uids), len(uids) <= bin_batch_num
    split_online_item_uids_click_rdd = online_item_uids_click_rdd.flatMap(split_item_users)

    # deterministic random partition to ensure the balance distribution of data
    # if using repartition API, which is non-deterministic. If one stage failed, the whole application will fail.
    split_online_item_uids_click_rdd = deterministic_random_partition(split_online_item_uids_click_rdd, partition_num=6000)

    # run to obtain [((u1, u2), [i1, i2, i3......]), ..., ], the commonly clicked item_list by u1, u2.
    # if u1,u2 only click one common item, then u1,u2 will contribute nothing to any item pairs, so we filter them out
    # This is also the limitation of swing: it will ignore item-pair that is clicked by ONLY one user or nobody
    # But this also helps to filter out some extremely unpopular items
    user_pair_common_click_item_list_rdd = split_online_item_uids_click_rdd.map(parse_user_pair).flatMap(lambda row: row) \
        .reduceByKey(lambda x, y: x + y).map(lambda row: (row[0], list(set(row[1]))))

    user_pair_common_click_item_list_rdd = user_pair_common_click_item_list_rdd.filter(lambda row: len(row[1]) > 1)
    # run to obtain [((i1, i2), sim), ... , ]
    item_pair_sim_rdd = user_pair_common_click_item_list_rdd.map(parse_item_pair)\
                                    .flatMap(lambda row: row).reduceByKey(lambda x, y: x + y)
    # map to str line split by blank space
    item_pair_swing_rdd = item_pair_sim_rdd.map(lambda row: row[0][0] + " " + row[0][1] + " " + str(row[1]))

    path = '%s/swing_daily_rdd/%s' % (s3_dir, date)
    item_pair_swing_rdd.coalesce(200).saveAsTextFile(path)  # combine small files, adjust the partition num first

    return 0


def start(date, days=1, env='dev'):
    print('date=%s, days=%s, env=%s' % (date, days, env))
    spark_session = SparkSession.builder.getOrCreate()
    ret = run(spark_session, date)
    if ret != 0:
        exit(-1)
    spark_session.stop()
    return ret


if __name__ == '__main__':
    fire.Fire(start)


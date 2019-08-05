import datetime
import os
import sys
import fire

os.environ['PYTHON_EGG_CACHE'] = '/tmp/.python-eggs'

reload(sys)
sys.setdefaultencoding('utf8')

from pyspark.sql import SparkSession

s3_dir = 's3://mx-machine-learning/dongxinzhou'


def read_online_item_pair_swing_daily_rdd(spark_session, date, days):
    # Violent modification
    if days == 0:
        days = 30
    file_list = []
    dt = datetime.datetime.strptime(str(date), '%Y%m%d')
    for i in range(days):
        new_dt = dt - datetime.timedelta(i)
        new_date = new_dt.strftime("%Y%m%d")
        file_dir = '%s/swing_daily_rdd/%s' % (s3_dir, new_date)
        file_list.append(file_dir)
    print(file_list)
    online_rdd = spark_session.sparkContext.textFile(','.join(file_list))
    return online_rdd


def output_line(row):
    if row[0][0] == '' or row[0][0] == ' ' or row[0][1] == '' or row[0][1] == ' ' or row[1] == 0:
        return []
    str_row = row[0][0] + " " + row[0][1] + " " + str(row[1])
    return [str_row]


def run(spark_session, date, days):
    online_item_pair_swing_daily_rdd = read_online_item_pair_swing_daily_rdd(spark_session, date, days)

    def obtain_item_pair_sw(row):
        row = row.strip().split()
        i1 = row[0]
        i2 = row[1]
        sw = float(row[2])  # important!!!
        if i1 > i2:
            i1, i2 = i2, i1
        return (i1, i2), sw

    online_item_pair_swing_daily_rdd = online_item_pair_swing_daily_rdd.map(obtain_item_pair_sw)
    merged_online_item_pair_swing_daily_rdd = online_item_pair_swing_daily_rdd.reduceByKey(lambda sw1, sw2: sw1 + sw2)

    # map to str line split by blank space
    merged_online_item_pair_swing_monthly_rdd = merged_online_item_pair_swing_daily_rdd.flatMap(output_line)

    path = '%s/merged_swing_rdd/%s' % (s3_dir, date)
    merged_online_item_pair_swing_monthly_rdd.coalesce(200).saveAsTextFile(path)
    return 0


def start(date, days=1, env='dev'):
    print('date=%s, days=%s, env=%s' % (date, days, env))
    spark_session = SparkSession.builder.getOrCreate()
    ret = run(spark_session, date, days)
    if ret != 0:
        exit(-1)
    spark_session.stop()
    return ret


if __name__ == '__main__':
    fire.Fire(start)



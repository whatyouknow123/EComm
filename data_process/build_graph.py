from pyspark import SparkContext, SparkConf, StorageLevel
import json

Index_User_Item = '../processed/index_train'
Graph_Saved = '../processed/graph_json'
Test = '../data/ECommAI_ubp_round1_test'
behavior = {'buy': '0', 'cart': '1', 'clk': '2', 'collect': '3'}


def map_line_u(line):
    # fields: 0:user id 1:item id 2: behavior 3: date
    fields = line.split()
    return int(fields[0]), (int(fields[1]), behavior[fields[2]])


def map_line_i(line):
    fields = line.split()
    return int(fields[1]), (int(fields[0]), behavior[fields[2]])


def to_block(k, v):
    block = {
        'node_id': k,
        'node_weight': 1.0,
        'uint64_feature': {},
        'float_feature': {},
        'binary_feature': {},
        'neighbor': {},
        'edge': []
    }
    for p in v:
        if p[1] not in block['neighbor']:
            block['neighbor'][p[1]] = {}
        block['neighbor'][p[1]][str(p[0])] = 1.0
        edge = {
            'src_id': k,
            'dst_id': p[0],
            'edge_type': int(p[1]),
            'weight': 1.0,
            'uint64_feature': {},
            'float_feature': {},
            'binary_feature': {}
        }
        block['edge'].append(edge)
    return block


def to_block_user(pair):
    block = to_block(pair[0], pair[1])
    block['node_type'] = 0
    return json.dumps(block)


def to_block_item(pair):
    block = to_block(pair[0], pair[1])
    block['node_type'] = 1
    return json.dumps(block)


# 构图
def build_graph_json(lines):
    test_uid = sc.textFile(Test)

    # 找出测试集中的冷启动用户
    cs_uid = test_uid.subtract(lines.map(lambda l: l.split()[0])).map(lambda x: int(x))

    u_i = lines.map(map_line_u)
    i_u = lines.map(map_line_i)

    u_is = u_i.combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)
    i_us = i_u.combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)

    # 找出度最大的Top50物品
    s_i_50 = sc.parallelize(i_us.map(lambda x: (len(x[1]), x[0])).sortByKey(False).top(50)).\
        map(lambda x: x[1])

    # 冷启动节点和Top50物品连接点击边
    u_o_is = cs_uid.cartesian(s_i_50).mapValues(lambda x: (x, '2')).\
        combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)
    # 反向连接
    i_o_us = s_i_50.cartesian(cs_uid).mapValues(lambda x: (x, '2')).\
        combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)

    # 合并
    u_is = u_is.union(u_o_is)
    i_us = i_us.union(i_o_us).reduceByKey(lambda x, y: x+y)

    u_block = u_is.map(to_block_user)
    i_block = i_us.map(to_block_item)

    all_block = u_block.union(i_block)

    all_block.saveAsTextFile(Graph_Saved)


# 统计图中的用户和物品
def get_ui_in_graph(lines):
    u = lines.map(lambda l: l.split()[0]).distinct()
    i = lines.map(lambda l: l.split()[1]).distinct()

    u.coalesce(1, True).saveAsTextFile('../processed/graph_user')
    i.coalesce(1, True).saveAsTextFile('../processed/graph_item')


# 统计节点度信息
def get_degree(lines):

    lines.flatMap(lambda l: l.split()[0:2]).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x+y).\
        map(lambda x: (x[1], 1)).reduceByKey(lambda x, y: x+y).sortByKey().\
        coalesce(1, True).saveAsTextFile('../processed/graph_degree')


if __name__ == "__main__":
    conf = SparkConf().setMaster("local").setAppName("EComm Data Process")
    sc = SparkContext(conf=conf)
    lines = sc.textFile(Index_User_Item)

    build_graph_json(lines)
    get_ui_in_graph(lines)
    get_degree(lines)

    sc.stop()

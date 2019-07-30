import os
import json

User_Feature = 'ECommAI_ubp_round1_user_feature'
Item_Feature = 'ECommAI_ubp_round1_item_feature'
User_Item = 'ECommAI_ubp_round1_train'
Test = 'ECommAI_ubp_round1_test'
Data_Path = '../data'
Out_Path = '../processed'


def generate_index():
    """
    输出文件：
        user_lookup: 用户编号 原始uid
        test_out_uid: 在测试文件中出现，但在用户资料中没有出现的用户（用户编号 原始uid）
        item_lookup: 物品编号 原始id
        index_train: ECommAI_ubp_round1_train中user id和item id转换为对应编号之后的记录，
                    丢弃了没有在用户资料和测试文件中出现的用户及没有在物品资料中出现的物品
        abnormal_uid: ECommAI_ubp_round1_train中没有在用户资料和测试文件中出现的用户的记录
        abnormal_iid: ECommAI_ubp_round1_train中没有在物品资料中出现的物品的记录
    """
    user_dict = {}
    item_dict = {}
    index = 0
    print('生成User Index...')
    with open(User_Feature, 'r') as f_u, open(os.path.join(Out_Path, 'user_lookup'), 'w') as o_f_u:
        for line in f_u:
            uid = int(line.split()[0])
            user_dict[uid] = index
            o_f_u.write('%d\t%d\n' % (index, uid))
            index += 1
        # 找出在测试文件中出现，但在用户资料中不存在的记录
        with open(Test, 'r') as o_f_tu, open(os.path.join(Out_Path, 'test_out_uid'), 'w') as o_f_tou:
            for line in o_f_tu:
                tuid = int(line.split()[0])
                if tuid not in user_dict:
                    user_dict[tuid] = index
                    o_f_u.write('%d\t%d\n' % (index, tuid))
                    o_f_tou.write('%d\t%d\n' % (index, tuid))
                    index += 1
                    print('Out Range User: %d' % tuid)
    print('生成Item Index...')
    with open(Item_Feature, 'r') as f_i, open(os.path.join(Out_Path, 'item_lookup'), 'w') as o_f_i:
        for line in f_i:
            iid = int(line.split()[0])
            item_dict[iid] = index
            o_f_i.write('%d\t%d\n' % (index, iid))
            index += 1
    print('生成User-Item...')
    with open(User_Item, 'r') as f_ui, open(os.path.join(Out_Path, 'index_train'), 'w') as o_f_ui, \
            open(os.path.join(Out_Path, 'abnormal_uid'), 'w') as o_f_au, \
            open(os.path.join(Out_Path, 'abnormal_iid'), 'w') as o_f_ai:
        abnormal_uid, abnormal_iid, normal_count = 0, 0, 0
        for line in f_ui:
            fields = line.split()
            uid, iid = int(fields[0]), int(fields[1])
            if uid not in user_dict:
                abnormal_uid += 1
                o_f_au.write(line)
                continue
            if iid not in item_dict:
                abnormal_iid += 1
                o_f_ai.write(line)
                continue
            fields[0] = str(user_dict[uid])
            fields[1] = str(item_dict[iid])
            o_f_ui.write('\t'.join(fields)+'\n')
            normal_count += 1
        print('正常记录：%d 用户缺失丢弃的记录：%d  物品缺失丢失的记录：%d' % (normal_count, abnormal_uid, abnormal_iid))


def generate_meta():
    print('生成元信息...')
    meta = {
        "node_type_num": 2,
        "edge_type_num": 4,
        "node_uint64_feature_num": 0,
        "node_float_feature_num": 0,
        "node_binary_feature_num": 0,
        "edge_uint64_feature_num": 0,
        "edge_float_feature_num": 0,
        "edge_binary_feature_num": 0
    }
    with open(os.path.join(Out_Path, 'meta'), 'w') as o_f_m:
        json.dump(meta, o_f_m, indent=4)


if __name__ == "__main__":
    User_Feature = os.path.join(Data_Path, User_Feature)
    Item_Feature = os.path.join(Data_Path, Item_Feature)
    User_Item = os.path.join(Data_Path, User_Item)
    Test = os.path.join(Data_Path, Test)
    generate_index()
    generate_meta()

import numpy as np
import faiss

embeddings = np.load('/mnt/d/Kikyou/CIKM 2019 EComm AI/tmp/processed/lite/embedding.npy')
d = embeddings.shape[1]
print(embeddings.shape)

index = faiss.IndexFlatL2(d)  # 创建索引
index.add(embeddings)  # 向索引中添加数据
print(index.ntotal)  # 索引中向量的数量

user_index = []
with open('/mnt/d/Kikyou/CIKM 2019 EComm AI/tmp/processed/lite/test_index', 'r') as f:
    for line in f:
        user_index.append(int(line))
user_lookup = {}
with open('/mnt/d/Kikyou/CIKM 2019 EComm AI/tmp/processed/lite/user_lookup', 'r') as f:
    for line in f:
        u = line.split()[1]
        i = int(line.split()[0])
        user_lookup[i] = u
item_lookup = {}
with open('/mnt/d/Kikyou/CIKM 2019 EComm AI/tmp/processed/lite/item_lookup', 'r') as f:
    for line in f:
        item = line.split()[1]
        i = int(line.split()[0])
        item_lookup[i] = item

user_embeddings = embeddings[user_index]
k = 1000
max_ui = 997406
D, I = index.search(user_embeddings, k)
user_recommendation = []
for i, u_i in enumerate(I):
    u = user_lookup[user_index[i]]
    recommendation = []
    for item in u_i:
        if item > max_ui:
            recommendation.append(item_lookup[item])
        if len(recommendation) >= 50: break
    if len(recommendation) < 50:
        print('k is not large enough')
        exit(-1)
    user_recommendation.append([u, recommendation])

with open('/mnt/d/Kikyou/CIKM 2019 EComm AI/tmp/processed/lite/recommendation', 'w') as f:
    for ui in user_recommendation:
        f.write('%s\t%s\n' % (ui[0], ','.join(ui[1])))
import json2dat
import os
import re
import multiprocessing

Graph_Json_Path = '../processed/graph_json'
Graph_Binary_Path = '../processed/graph_binary'
Meta_File = '../processed/meta'
Thread_Count = 8


def convert(graph_files, meta_info, out_path):
    for gf in graph_files:
        filename, _ = os.path.splitext(os.path.split(gf)[1])
        output_file = os.path.join(out_path, filename + '.dat')
        if os.path.exists(output_file):
            print("Skip: %s" % output_file)
        else:
            c = json2dat.Converter(meta_info, gf, output_file+'.tmp')
            c.do()
            os.rename(output_file+'.tmp', output_file)
    print(multiprocessing.current_process().name + ' Finished')


def get_graph_files(path):
    graph_files = []
    re_gf = re.compile("part-\d+")
    for root, _, files in os.walk(path):
        for name in files:
            m = re_gf.match(name)
            if not m: continue
            if m.end(0) == len(name):
                graph_files.append(os.path.join(root, name))
    return graph_files


if __name__ == '__main__':
    gf = get_graph_files(Graph_Json_Path)
    c = len(gf)
    t_c = int(c / Thread_Count)
    tasks = []
    for i in range(0, c, t_c):
        p_gf = gf[i:i+t_c]
        t = multiprocessing.Process(name='Task-%d-%d'% (i, i+len(p_gf)-1),
                                    target=convert, args=(p_gf, Meta_File, Graph_Binary_Path,))
        tasks.append(t)
        t.start()
    for t in tasks:
        t.join()

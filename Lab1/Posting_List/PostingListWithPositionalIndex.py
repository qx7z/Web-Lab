import glob
import math
import os

def construct_postinglist_with_positional_index(normalized_txt_dir):
    full_pattern = os.path.join(normalized_txt_dir, "*")

    txt_files = glob.glob(full_pattern)

    filename_to_ID = {}
    postinglist = {}

    ID = 1
    for file in txt_files:
        base = os.path.basename(file)
        filename_to_ID[base] = ID
        ID += 1

    for file in txt_files:
        tokens = []
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            tokens = content.strip().split()
        base = os.path.basename(file)
        print(f"正在处理: {base}"+ f"词项数: {len(tokens)}")
        position_index = 0
        for token in tokens:
            if token not in postinglist.keys():
                postinglist[token] = []
                token_positional_index_list = []
                token_positional_index_list.append(position_index)
                postinglist[token].append((filename_to_ID[base],token_positional_index_list))  
            else:
                if filename_to_ID[base] != postinglist[token][-1][0]: # 确保文档ID唯一
                    token_positional_index_list = []
                    token_positional_index_list.append(position_index)
                    postinglist[token].append((filename_to_ID[base],token_positional_index_list))  
                else:
                    postinglist[token][-1][1].append(position_index)
            
            position_index += 1 
    
    print(f"一共 {len(txt_files)} 个txt文档。")
    print(f"一共 {len(postinglist.keys())} 种token。")

    return filename_to_ID, postinglist

def add_skip_pointers(postinglist):
    for token, IDList in postinglist.items():
        stride = math.floor(math.sqrt(len(IDList)))
        IDList_with_pointers = []
        for i,ID_with_position_index in enumerate(IDList):
            if i % stride == 0 and i + stride < len(IDList):
                IDList_with_pointers.append((ID_with_position_index,IDList[i + stride]))
            else:
                IDList_with_pointers.append((ID_with_position_index,None))
        postinglist[token] = IDList_with_pointers
    
    return postinglist

def blocking_compression(postinglist):
    DicStr = ""
    for token in postinglist.key():
        DicStr += token
    tokenp_len = math.ceil(math.log(len(DicStr),2))
    blocking_compressed_postinglist = {}
    
    

if __name__ == "__main__":
    normalized_txt_dir = os.path.join(os.getcwd(),"Lab1" ,"XML_Parsing_And_Normalization","Normalized_txt_files" )
    filename_to_ID, postinglist = construct_postinglist_with_positional_index(normalized_txt_dir)
    postinglist_with_skip_pointers = add_skip_pointers(postinglist)
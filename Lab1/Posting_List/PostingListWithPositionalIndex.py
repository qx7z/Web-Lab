import glob
import math
import os
import pickle
import sys
from pympler import asizeof

# 构建含位置信息的倒排表
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
            if(len(token) > 255):
                continue
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

# 添加跳表
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

# 分离索引和倒排表内容，分开存储
def separate_index_and_postings_with_df_in_memory(postinglist, postings_filename="postings_df_mem.bin", index_filename="term_dictionary.pkl"):
    """
    分离倒排表。
    将 DF 和磁盘指针 (offset, size) 存入内存中的词项字典。
    只将倒排列表本身写入磁盘文件。
    
    :return: 内存中的词项字典 {token: (df, offset, size)}。
    """
    print("--- 开始分离索引与内容 (DF存入内存) ---")
    term_dictionary_in_memory = {}

    output_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    output_bin_path = os.path.join(output_dir,postings_filename)
    with open(output_bin_path, "wb") as f_postings:
        for token, postings_data_list in postinglist.items():
            doc_frequency = len(postings_data_list)
            offset = f_postings.tell()
            serialized_data = pickle.dumps(postings_data_list)
            f_postings.write(serialized_data)
            size = len(serialized_data)
            term_dictionary_in_memory[token] = (doc_frequency, offset, size)

    output_dic_path = os.path.join(output_dir,index_filename)
    with open(output_dic_path, "wb") as f_index:
        pickle.dump(term_dictionary_in_memory, f_index)
    
    print("内存中的词项字典 (含DF) 构建完成。")
    return term_dictionary_in_memory

# 使用索引从倒排表中获得倒排表内容
def retrieve_postings_list_only(token, term_dict, postings_filename):
    if token not in term_dict:
        return None
    
    df, offset, size = term_dict[token]
    
    with open(postings_filename, "rb") as f:
        f.seek(offset)
        serialized_data = f.read(size)
        postings_list = pickle.loads(serialized_data)
        return postings_list

# 加载索引
def load_term_dic(term_dic_path):
    with open(term_dic_path, "rb") as f_index:
        loaded_term_dictionary = pickle.load(f_index)
    return loaded_term_dictionary

def compress_as_string(term_dict):
    """
    使用 Dictionary-as-a-String 方法压缩词典。
    
    :param term_dict: 内存中的词项字典 {token: (df, offset, size)}。
    :return: 一个元组，包含:
             - concatenated_string: 所有词项拼接成的大字符串
             - metadata_list: 每个词项的元数据列表 [(df, offset, size, term_pointer), ...]
    """
    print("\n--- 开始使用 'Dictionary-as-a-String' 方法压缩 ---")
    if not term_dict:
        return "", []

    sorted_tokens = sorted(term_dict.keys())
    
    string_builder = []
    metadata_list = []
    current_term_pointer = 0
    
    for token in sorted_tokens:
        df, offset, size = term_dict[token]
        metadata_list.append((df, offset, size, current_term_pointer))
        string_builder.append(token)
        current_term_pointer += len(token)

    concatenated_string = "".join(string_builder)
    
    print("索引压缩完成。")
    return concatenated_string, metadata_list

# 按块编码
def compress_with_blocking(term_dict, k=4):
    """
    使用按块存储 (Blocking) 方法压缩词典。
    
    :param term_dict: 内存中的词项字典 {token: (df, offset, size)}。
    :param k: 块大小，即每k个词项存储一个指针。
    :return: 一个元组，包含:
             - block_string: 按 <len><term>... 格式拼接的块字符串
             - block_metadata: 每个块的元数据 [(first_term_in_block, block_pointer), ...]
             - full_sorted_metadata: 所有词项的完整元数据，按词项排序
    """
    print(f"\n--- 开始使用 '按块存储 (Blocking)' 方法压缩 (k={k}) ---")
    if not term_dict:
        return "", [], []

    sorted_tokens = sorted(term_dict.keys())
    
    string_builder = []
    block_metadata = []         # 存储每个块的第一个词和块指针
    full_sorted_metadata = []   # 存储所有词项的元数据
    current_char_offset = 0
    
    for i in range(0, len(sorted_tokens)):
        token = sorted_tokens[i]
        
        full_sorted_metadata.append(term_dict[token])
        
        if i % k == 0:
            block_metadata.append((token, current_char_offset))

        if len(token) > 255:
            raise ValueError(f"Token '{token}' is too long for single-byte length storage.")    
        string_builder.append(chr(len(token)))
        string_builder.append(token)
        
        current_char_offset += 1 + len(token)

    block_string = "".join(string_builder)
    
    print("索引压缩完成。")
    return block_string, block_metadata, full_sorted_metadata

# 使用按块编码的查询
def find_token_in_blocked_dict(query_token, block_string, block_meta, full_meta, k):
    """
    在分块结构中查找词项。
    
    :return: 如果找到，返回元数据 (df, offset, size)；否则返回 None。
    """
    low = 0
    high = len(block_meta) - 1
    target_block_index = -1

    # 二分查找找query所属快
    while low <= high:
        mid = (low + high) // 2
        first_term_in_block = block_meta[mid][0]
        
        if first_term_in_block <= query_token:
            target_block_index = mid
            low = mid + 1
        else:
            high = mid - 1
            
    if target_block_index == -1:
        return None 

    block_start_char_ptr = block_meta[target_block_index][1]
    
    if target_block_index + 1 < len(block_meta):
        block_end_char_ptr = block_meta[target_block_index + 1][1]
    else:
        block_end_char_ptr = len(block_string)

    current_pos = block_start_char_ptr
    term_index_in_block = 0
    
    while current_pos < block_end_char_ptr:

        term_len = ord(block_string[current_pos])
        
        term = block_string[current_pos + 1 : current_pos + 1 + term_len]

        if term == query_token:
            global_index = (target_block_index * k) + term_index_in_block
            return full_meta[global_index]

        current_pos += 1 + term_len
        term_index_in_block += 1
        
    return None 

# 前端编码 :成对前缀编码法在压缩率和稳定性上都全面优于块级公共前缀法
def find_common_prefix_len(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def compress_blocking_with_front_coding(term_dict, k=4):
    """
    结合按块存储和前端编码进行压缩。
    """
    print(f"\n--- 结合 '按块存储' 与 '前端编码'进行压缩 (k={k}) ---")
    if not term_dict:
        return "", [], []

    sorted_tokens = sorted(term_dict.keys())
    
    string_builder = []
    block_metadata = []
    full_sorted_metadata = [term_dict[token] for token in sorted_tokens]
    current_char_offset = 0

    for i in range(0, len(sorted_tokens), k):
        block_tokens = sorted_tokens[i : i + k]
        
        if not block_tokens: continue

        first_term_in_block = block_tokens[0]
        block_metadata.append((first_term_in_block, current_char_offset))

        string_builder.append(first_term_in_block)
        current_char_offset += len(first_term_in_block)
        
        for j in range(1, len(block_tokens)):
            previous_token = block_tokens[j-1]
            current_token = block_tokens[j]

            prefix_len = find_common_prefix_len(previous_token, current_token)
            suffix = current_token[prefix_len:]
            suffix_len = len(suffix)

            if prefix_len > 255 or suffix_len > 255:
                raise ValueError("Prefix or suffix too long for single-byte storage.")

            string_builder.append(chr(prefix_len))
            string_builder.append(chr(suffix_len))
            string_builder.append(suffix)
            
            current_char_offset += 2 + suffix_len # 2 bytes for lengths + suffix

    block_string = "".join(string_builder)
    
    print("索引压缩完成。")
    return block_string, block_metadata, full_sorted_metadata

# 使用前端编码的查询
def find_token_in_blocked_front_coded_dict(query_token, block_string, block_meta, full_meta, k):
    """
    在前端编码的分块结构中查找词项。
    
    :return: 如果找到，返回元数据 (df, offset, size)；否则返回 None。
    """
    low, high = 0, len(block_meta) - 1
    target_block_index = -1
    while low <= high:
        mid = (low + high) // 2
        if block_meta[mid][0] <= query_token:
            target_block_index = mid
            low = mid + 1
        else:
            high = mid - 1
    if target_block_index == -1: return None

    reconstructed_term = block_meta[target_block_index][0]
    
    term_index_in_block = 0
    if reconstructed_term == query_token:
        global_index = (target_block_index * k) + term_index_in_block
        return full_meta[global_index]
        
    current_pos = block_meta[target_block_index][1] + len(reconstructed_term)
    
    block_end_pos = len(block_string)
    if target_block_index + 1 < len(block_meta):
        block_end_pos = block_meta[target_block_index + 1][1]
    
    term_index_in_block += 1
    while current_pos < block_end_pos:
        # 解码 <prefix_len><suffix_len><suffix>
        prefix_len = ord(block_string[current_pos])
        suffix_len = ord(block_string[current_pos + 1])
        suffix = block_string[current_pos + 2 : current_pos + 2 + suffix_len]
        reconstructed_term = reconstructed_term[:prefix_len] + suffix
        
        # 比较
        if reconstructed_term == query_token:
            global_index = (target_block_index * k) + term_index_in_block
            return full_meta[global_index]
            
        # 准备下一次迭代
        current_pos += 2 + suffix_len
        term_index_in_block += 1
        
    return None

def get_deep_sizeof(o, ids):
    """
    递归地计算一个对象的深度内存大小。
    'ids' 是一个 set，用于防止重复计算同一个对象。
    """
    # 如果对象已经被计算过，直接返回0，避免循环引用导致的死循环
    if id(o) in ids:
        return 0

    # 获取对象的浅层大小
    size = sys.getsizeof(o)
    ids.add(id(o))

    # 如果是字典，递归计算所有键和值的大小
    if isinstance(o, dict):
        size += sum(get_deep_sizeof(k, ids) + get_deep_sizeof(v, ids) for k, v in o.items())
    # 如果是其他容器（列表、元组、集合等），递归计算所有元素的大小
    elif isinstance(o, (list, tuple, set)):
        size += sum(get_deep_sizeof(i, ids) for i in o)

    return size

if __name__ == "__main__":
    normalized_txt_dir = os.path.join(os.getcwd(),"Lab1" ,"XML_Parsing_And_Normalization","Normalized_txt_files" )
    filename_to_ID, postinglist = construct_postinglist_with_positional_index(normalized_txt_dir)
    postinglist_with_skip_pointers = add_skip_pointers(postinglist)
    term_dictionary = separate_index_and_postings_with_df_in_memory(postinglist_with_skip_pointers, "postings_df_mem.bin","term_dictionary.pkl")

    block_string, block_meta, full_meta = compress_with_blocking(term_dictionary, k=4)
    blocking_structure_to_save = (block_string, block_meta, full_meta)

    block_string_fc, block_meta_fc, full_meta_fc = compress_blocking_with_front_coding(term_dictionary, k=4)
    blocking_fc_structure_to_save = (block_string_fc, block_meta_fc, full_meta_fc)

    # 计算索引大小
    index_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    # 原始索引
    file1 = "index_original.bin"
    file1_path = os.path.join(index_dir,file1)
    with open(file1_path, "wb") as f:
        pickle.dump(term_dictionary, f)
    #size_original_file = os.path.getsize(file1_path)
    #size_original_file = get_deep_sizeof(term_dictionary,set())
    size_original_file = asizeof.asizeof(term_dictionary)

    # 按块存储
    file2 = "index_blocking.bin"
    file2_path = os.path.join(index_dir,file2)
    with open(file2_path, "wb") as f:
        pickle.dump(blocking_structure_to_save, f)
    #size_blocking_file = os.path.getsize(file2_path)
    #size_blocking_file = get_deep_sizeof(blocking_structure_to_save,set())
    size_blocking_file = asizeof.asizeof(blocking_structure_to_save)


    # 前端编码
    file3 = "index_front_coding.bin"
    file3_path = os.path.join(index_dir,file3)
    with open(file3_path, "wb") as f:
        pickle.dump(blocking_fc_structure_to_save, f)
    # size_fc_file = os.path.getsize(file3_path)
    # size_fc_file = get_deep_sizeof(blocking_fc_structure_to_save,set())
    size_fc_file = asizeof.asizeof(blocking_fc_structure_to_save)
    

    # --- 3. 总结与比较 ---
    print("="*60)
    print("          索引文件大小比较总结")
    print("="*60)
    print(f"{'索引类型':<30} | {'文件名':<25} | {'大小':>12}")
    print("-"*65)
    print(f"{'1.原始索引':<30} | {file1:<25} | {size_original_file:>12}")

    # 计算压缩率
    # 相对于传统的定长存储模型，Python的动态对象模型本身就是一种高度优化的、基于指针的“压缩”存储方案。它从根本上避免了因定长预分配而产生的巨大空间浪费。
    blocking_compression = (1 - size_blocking_file / size_original_file) * 100
    print(f"{'2.按块存储':<30} | {file2:<25} | {size_blocking_file:>12} ({blocking_compression:.1f}% 压缩)")
   
    fc_compression = (1 - size_fc_file / size_original_file) * 100
    print(f"{'3.前端编码':<30} | {file3:<25} | {size_fc_file:>12} ({fc_compression:.1f}% 压缩)")
    print("="*65)

    query = "python"
    p1 = term_dictionary[query]

    p2 = find_token_in_blocked_dict(query,block_string, block_meta, full_meta,4)

    p3 = find_token_in_blocked_front_coded_dict(query,block_string_fc, block_meta_fc, full_meta_fc,4)

    assert p1 == p2,"find_token_in_blocked_dict有误，无法从按块存储的索引中获取倒排表内容"
    assert p1 == p3,"find_token_in_front_coded_blocked_dict有误，无法从前端编码的索引中获取倒排表内容"

    print("三种索引均构建成功!")
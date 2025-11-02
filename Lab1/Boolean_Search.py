from math import floor, sqrt
import os
import time
from pympler import asizeof
from Posting_List.PostingListWithPositionalIndex import load_term_dic,load_filename_to_ID,retrieve_postings_list_only
from Posting_List.PostingListWithPositionalIndex import compress_with_blocking,compress_blocking_with_front_coding
from Posting_List.PostingListWithPositionalIndex import find_token_in_blocked_dict,find_token_in_blocked_front_coded_dict

def retrieve_posting_list_entry(entry,term_dic):
    postinglist_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    postinglist_file = "postings_df_mem.bin"
    return retrieve_postings_list_only(entry,term_dic,os.path.join(postinglist_dir,postinglist_file))

def intersect_postings(p1, p2):
    result = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        docID1 = p1[i][0]
        docID2 = p2[j][0]
        
        if docID1 == docID2:
            result.append(p1[i])
            i += 1
            j += 1
        elif docID1 < docID2:
            i += 1
        else: 
            j += 1
            
    return result

def union_postings(p1, p2):
    result = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        docID1 = p1[i][0]
        docID2 = p2[j][0]
        
        if docID1 == docID2:
            result.append(p1[i]) 
            j += 1
        elif docID1 < docID2:
            result.append(p1[i])
            i += 1
        else: 
            result.append(p2[j])
            j += 1
            
    while i < len(p1):
        result.append(p1[i])
        i += 1
    while j < len(p2):
        result.append(p2[j])
        j += 1
        
    return result

def intersect_positions_with_distance(pos_list1, pos_list2, distance):
    i, j = 0, 0
    while i < len(pos_list1) and j < len(pos_list2):
        pos1 = pos_list1[i]
        pos2 = pos_list2[j]
        
        if pos2 == pos1 + distance:
            return True 
        elif pos2 > pos1 + distance:
            i += 1
        else:
            j += 1
    return False

def intersect_postings_with_pos(p1, p2):
    result = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        docID1 = p1[i][0]
        docID2 = p2[j][0]
        
        if docID1 == docID2:
            result.append((p1[i], p2[j]))
            i += 1
            j += 1
        elif docID1 < docID2:
            i += 1
        else: 
            j += 1
            
    return result

def intersect_with_skips(p1_with_skips, p2_with_skips):
    p1, skips1 = p1_with_skips
    p2, skips2 = p2_with_skips
    result = []
    i, j = 0, 0
    comparisons = 0
    
    while i < len(p1) and j < len(p2):
        comparisons += 1
        docID1 = p1[i][0]
        docID2 = p2[j][0]

        if docID1 == docID2:
            result.append(p1[i])
            i += 1
            j += 1
        elif docID1 < docID2:
            if docID1 in skips1 and p1[skips1[docID1]][0] <= docID2:
                i = skips1[docID1]
            else:
                i += 1
        else: 
            if docID2 in skips2 and p2[skips2[docID2]][0] <= docID1:
                j = skips2[docID2]
            else:
                j += 1
                
    return (result, comparisons)

def build_skip_list(postings_list, step):
    skip_list = {}
    for i in range(0, len(postings_list), step):
        if i + step < len(postings_list):
            from_docID = postings_list[i][0]
            to_docID_index = i + step
            skip_list[from_docID] = to_docID_index
            
    return skip_list

if __name__ == "__main__":
    postinglist_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    index_file = "index_original.bin"
    filename_to_ID_file = "filename_to_ID.bin"
    term_dic = load_term_dic(os.path.join(postinglist_dir,index_file))
    filename_to_ID = load_filename_to_ID(os.path.join(postinglist_dir,filename_to_ID_file))
    # 5.1 query1
    query_terms = ["system", "program","python"]

    print(f"\n执行查询: '{' AND '.join(query_terms)}'")
    for term in query_terms:
        print(f" - 文档频率 (DF) of '{term}': {term_dic[term][0]}")

    #按字母序
    print("\n--- 测试直接顺序: (program AND python) AND system ---")

    p_prog = retrieve_posting_list_entry("program", term_dic)
    p_py   = retrieve_posting_list_entry("python",  term_dic)
    p_sys  = retrieve_posting_list_entry("system",  term_dic)

    start_time = time.perf_counter()

    intermediate_result1 = intersect_postings(p_prog, p_sys)
    final_result1 = intersect_postings(intermediate_result1, p_py)

    end_time = time.perf_counter()
    duration1 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_prog)} ∩ {len(p_sys)}) = {len(intermediate_result1)}")
    print(f"最终结果: {[item[0] for item in final_result1]}")
    print(f"执行时间: {duration1:.2f} 微秒 (μs)")


    # 最优
    print("\n--- 测试最优顺序: program AND (system AND python) ---")

    start_time = time.perf_counter()

    intermediate_result2 = intersect_postings(p_py, p_sys)
    final_result2 = intersect_postings(intermediate_result2, p_prog)

    end_time = time.perf_counter()
    duration2 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_py)} ∩ {len(p_sys)}) = {len(intermediate_result2)}")
    print(f"最终结果: {[item[0] for item in final_result2]}")
    print(f"执行时间: {duration2:.2f} 微秒 (μs)")


    # 5.1 query2
    query_terms = ["next","paper","apples"]

    print(f"\n执行查询: '{' AND '.join(query_terms)}'")
    for term in query_terms:
        print(f" - 文档频率 (DF) of '{term}': {term_dic[term][0]}")

    #按字母序
    print("\n--- 测试直接顺序: (next AND paper) AND apples ---")

    p_nex = retrieve_posting_list_entry("next", term_dic)
    p_pap = retrieve_posting_list_entry("paper",  term_dic)
    p_app = retrieve_posting_list_entry("apples",  term_dic)

    start_time = time.perf_counter()

    intermediate_result1 = intersect_postings(p_nex, p_pap)
    final_result1 = intersect_postings(intermediate_result1, p_app)

    end_time = time.perf_counter()
    duration1 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_nex)}  ∩ {len(p_pap)}) = {len(intermediate_result1)}")
    print(f"最终结果: {[item[0] for item in final_result1]}")
    print(f"执行时间: {duration1:.2f} 微秒 (μs)")


    # 最优
    print("\n--- 测试最优顺序: next AND (paper AND apple) ---")

    start_time = time.perf_counter()

    intermediate_result2 = intersect_postings(p_pap, p_app)
    final_result2 = intersect_postings(intermediate_result2, p_nex)

    end_time = time.perf_counter()
    duration2 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_pap)} ∩ {len(p_app)}) = {len(intermediate_result2)}")
    print(f"最终结果: {[item[0] for item in final_result2]}")
    print(f"执行时间: {duration2:.2f} 微秒 (μs)")

    # 5.1 query3
    query_terms = ["next","paper","apples"]

    print(f"\n执行查询: '{' OR '.join(query_terms)}'")
    for term in query_terms:
        print(f" - 文档频率 (DF) of '{term}': {term_dic[term][0]}")

    #按字母序
    print("\n--- 测试直接顺序: (next OR paper) OR apples ---")

    p_nex = retrieve_posting_list_entry("next", term_dic)
    p_pap = retrieve_posting_list_entry("paper",  term_dic)
    p_app = retrieve_posting_list_entry("apples",  term_dic)

    start_time = time.perf_counter()

    intermediate_result1 = union_postings(p_nex, p_pap)
    final_result1 = union_postings(intermediate_result1, p_app)

    end_time = time.perf_counter()
    duration1 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_nex)} U {len(p_pap)}) = {len(intermediate_result1)}")
    print(f"最终结果: {len(final_result1)}")
    print(f"执行时间: {duration1:.2f} 微秒 (μs)")


    # 最优
    print("\n--- 测试最优顺序: next OR (paper OR apple) ---")

    start_time = time.perf_counter()

    intermediate_result2 = union_postings(p_pap, p_app)
    final_result2 = union_postings(intermediate_result2, p_nex)

    end_time = time.perf_counter()
    duration2 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_pap)} U {len(p_app)}) = {len(intermediate_result2)}")
    print(f"最终结果: {len(final_result2)}")
    print(f"执行时间: {duration2:.2f} 微秒 (μs)")

    print("\n--- 测试顺序: paper OR (next OR apple) ---")

    start_time = time.perf_counter()

    intermediate_result2 = union_postings(p_nex, p_app)
    final_result2 = union_postings(intermediate_result2, p_pap)

    end_time = time.perf_counter()
    duration2 = (end_time - start_time) * 1_000_000 # 转换为微秒

    print(f"中间结果大小: len({len(p_nex)} U {len(p_app)}) = {len(intermediate_result2)}")
    print(f"最终结果: {len(final_result2)}")
    print(f"执行时间: {duration2:.2f} 微秒 (μs)")

    #5.2
    block_string, block_meta, full_meta = compress_with_blocking(term_dic, k=4)
    block_string_fc, block_meta_fc, full_meta_fc = compress_blocking_with_front_coding(term_dic, k=4)
    query_terms = ["system","python","next","paper","apple","program"]
    iterations = 1000
    # 原始索引
    start_time = time.perf_counter()
    for _ in range(iterations):
        for term in query_terms:
            p = term_dic[term]
    end_time = time.perf_counter()
    duration_original = (end_time - start_time)
    avg_time_original = (duration_original / (iterations * len(query_terms))) * 1_000_000_000

    print(f"\n--- 1. 使用原始索引 ---")
    print(f"总耗时: {duration_original:.6f} 秒")
    print(f"平均每次词项查找耗时: {avg_time_original:.2f} 纳秒 (ns)")

    # 按块编码
    start_time = time.perf_counter()
    for _ in range(iterations):
        for term in query_terms:
            p = find_token_in_blocked_dict(term,block_string, block_meta, full_meta,4)
    end_time = time.perf_counter()
    duration_original = (end_time - start_time)
    avg_time_original = (duration_original / (iterations * len(query_terms))) * 1_000_000_000

    print(f"\n--- 2. 使用按块存储 ---")
    print(f"总耗时: {duration_original:.6f} 秒")
    print(f"平均每次词项查找耗时: {avg_time_original:.2f} 纳秒 (ns)")

    # 前端编码
    start_time = time.perf_counter()
    for _ in range(iterations):
        for term in query_terms:
            p = find_token_in_blocked_front_coded_dict(term,block_string_fc, block_meta_fc, full_meta_fc,4)
    end_time = time.perf_counter()
    duration_original = (end_time - start_time)
    avg_time_original = (duration_original / (iterations * len(query_terms))) * 1_000_000_000

    print(f"\n--- 3. 使用前端编码 ---")
    print(f"总耗时: {duration_original:.6f} 秒")
    print(f"平均每次词项查找耗时: {avg_time_original:.2f} 纳秒 (ns)")

    # 5.3
    phrase = "weather permit"

    terms = phrase.split()

    term1, term2 = terms[0], terms[1]
    
    p1 = retrieve_posting_list_entry(term1, term_dic)
    p2 = retrieve_posting_list_entry(term2, term_dic)

    print("\n查询 \"weather AND permit\"")
    merged_docs = intersect_postings(p1,p2)
    merged_docs = [docID[0] for docID in merged_docs]
    print(f"找到{len(merged_docs)}个同时含\"weather\"和\"permit\"的文档")
    # print(merged_docs)

    candidate_pairs = intersect_postings_with_pos(p1, p2)
    
    final_docs = []
    for entry1, entry2 in candidate_pairs:
        docID = entry1[0]
        pos_list1 = entry1[1]
        pos_list2 = entry2[1]
        
        if intersect_positions_with_distance(pos_list1, pos_list2, 1):
            final_docs.append(docID)

    reverse_dict = {v: k for k, v in filename_to_ID.items()}

    print("\n查询 \"weather permit\"")
    # final_docs = [reverse_dict[docID] for docID in final_docs]
    print(f"找到{len(final_docs)}个含\"weather permit\"短语的文档")
    # print(final_docs)
    print(f"前7个文档名为{[reverse_dict[docID] for docID in final_docs[:7]]}")

    # 5.4
    query_terms = [("system","python"),("next","paper"),("apple","program")]
    terms_df =[]
    p_no_skips_size = 0
    p_with_skips_10_size = 0
    p_with_skips_50_size = 0
    p_with_skips_sqrtl_size = 0
    print("\n--- A. 检索效率分析 ---")
    # print(f"{query_terms}的DF为{terms_df}")

    for terms in query_terms:
        term1,term2 = terms
        p1 = retrieve_posting_list_entry(term1, term_dic)
        p2 = retrieve_posting_list_entry(term2, term_dic)
        terms_df.append((len(p1),len(p2)))
        print(f"terms:{terms},terms_df{(len(p1),len(p2))}")

        p1_no_skips = (p1, {})
        p2_no_skips = (p2, {})
        p_no_skips_size += asizeof.asizeof(p1_no_skips)
        p_no_skips_size += asizeof.asizeof(p2_no_skips)
        start_time = time.perf_counter()
        result1, comps1 = intersect_with_skips(p1_no_skips, p2_no_skips)
        duration1 = (time.perf_counter() - start_time) * 1_000_000
        print(f"【无跳表】: 找到 {len(result1)} 个结果, 进行了 {comps1} 次比较, 耗时 {duration1:.2f} μs")


        p1_with_skips_10 = (p1,build_skip_list(p1,10))
        p2_with_skips_10 = (p2,build_skip_list(p2,10))
        p_with_skips_10_size += asizeof.asizeof(p1_with_skips_10)
        p_with_skips_10_size += asizeof.asizeof(p2_with_skips_10)
        start_time = time.perf_counter()
        result2, comps2 = intersect_with_skips(p1_with_skips_10, p2_with_skips_10)
        duration2 = (time.perf_counter() - start_time) * 1_000_000
        print(f"【步长 k=10】: 找到 {len(result2)} 个结果, 进行了 {comps2} 次比较, 耗时 {duration2:.2f} μs")

        p1_with_skips_50 = (p1,build_skip_list(p1,50))
        p2_with_skips_50 = (p2,build_skip_list(p2,50))
        p_with_skips_50_size += asizeof.asizeof(p1_with_skips_50)
        p_with_skips_50_size += asizeof.asizeof(p2_with_skips_50)
        start_time = time.perf_counter()
        result3, comps3 = intersect_with_skips(p1_with_skips_50, p2_with_skips_50)
        duration3 = (time.perf_counter() - start_time) * 1_000_000
        print(f"【步长 k=50】: 找到 {len(result3)} 个结果, 进行了 {comps3} 次比较, 耗时 {duration3:.2f} μs")

        p1_with_skips_sqrtl = (p1,build_skip_list(p1,floor(sqrt(len(p1)))))
        p2_with_skips_sqrtl = (p2,build_skip_list(p2,floor(sqrt(len(p2)))))
        p_with_skips_sqrtl_size += asizeof.asizeof(p1_with_skips_sqrtl)
        p_with_skips_sqrtl_size += asizeof.asizeof(p2_with_skips_sqrtl)
        start_time = time.perf_counter()
        result4, comps4 = intersect_with_skips(p1_with_skips_sqrtl, p2_with_skips_sqrtl)
        duration4 = (time.perf_counter() - start_time) * 1_000_000
        print(f"【步长 k=sqrtl】: 找到 {len(result4)} 个结果, 进行了 {comps4} 次比较, 耗时 {duration4:.2f} μs")

    print("\n--- B. 存储性能分析 ---")
    print(f"无跳表索引大小: {p_no_skips_size}")
    print(f"步长 k=10 索引大小: {p_with_skips_10_size}")
    print(f"步长 k=50 索引大小: {p_with_skips_50_size}")
    print(f"步长 k=sqrtl 索引大小: {p_with_skips_sqrtl_size}")
    
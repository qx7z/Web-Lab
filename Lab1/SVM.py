from collections import defaultdict
import math
import os
import numpy as np
from Posting_List.PostingListWithPositionalIndex import load_term_dic,load_filename_to_ID,retrieve_postings_list_only
from Boolean_Search import retrieve_posting_list_entry

def compute_idf(term_dict, N):
    idf_scores = {}
    for token in term_dict.keys():
        df = term_dict[token][0]
        idf_scores[token] = math.log10(N / df)
    return idf_scores

def compute_doc_vectors(term_dict, idf):
    doc_vectors = defaultdict(dict)
    for token in term_dict.keys():
        p = retrieve_posting_list_entry(token, term_dic)
        for (docID, index) in p:
            tf_score = 1 + math.log10(len(index))
            tf_idf_score = tf_score * idf[token]
            doc_vectors[docID][token] = tf_idf_score
    return doc_vectors

def cosine_similarity(vec1, vec2, term_dict):

    v1 = np.array([vec1.get(token, 0) for token in term_dict.keys()])
    v2 = np.array([vec2.get(token, 0) for token in term_dict.keys()])
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    
    return dot_product / (norm_v1 * norm_v2)


def search_vsm(query, doc_vectors, idf, term_dict):
    query_tokens = query.lower().split()
    query_counts = defaultdict(int)
    for token in query_tokens:
        query_counts[token] += 1
    
    query_vector = {}
    for token, count in query_counts.items():
        tf_score = 1 + math.log10(count)
        idf_score = idf.get(token, 0) 
        query_vector[token] = tf_score * idf_score
            
    scores = {}
    i = 0
    total_doc_N = len(doc_vectors)
    for docID, doc_vec in doc_vectors.items():
        i += 1
        print(f"compute the similarity of query and {docID},{i}/{total_doc_N}")
        scores[docID] = cosine_similarity(query_vector, doc_vec, term_dict)
        
    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_docs

if __name__ == "__main__":
    postinglist_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    index_file = "index_original.bin"
    filename_to_ID_file = "filename_to_ID.bin"
    term_dic = load_term_dic(os.path.join(postinglist_dir,index_file))
    filename_to_ID = load_filename_to_ID(os.path.join(postinglist_dir,filename_to_ID_file))
    reverse_dict = {v: k for k, v in filename_to_ID.items()}
    total_doc_N = len(filename_to_ID)

    idf_scores = compute_idf(term_dic,total_doc_N)
    doc_vectors = compute_doc_vectors(term_dic,idf_scores)
    query = "weather permit"
    ranked_results = search_vsm(query,doc_vectors,idf_scores,term_dic)
    topk = 7
    i = 0
    for docID, score in ranked_results:
        if(i < topk):
            i += 1
            print(f"评分第{i}高的文档 {docID}: Score = {score:.4f} |文档名: '{reverse_dict[docID]}'")
        else:
            break
    # vec1 = doc_vectors[30025]
    # vec2 = doc_vectors[36376]
    # print(doc_vectors[30025])
    # print(doc_vectors[36376])
    # query_tokens = query.lower().split()
    # query_counts = defaultdict(int)
    # for token in query_tokens:
    #     query_counts[token] += 1
    
    # query_vector = {}
    # for token, count in query_counts.items():
    #     tf_score = 1 + math.log10(count)
    #     idf_score = idf_scores.get(token, 0) 
    #     query_vector[token] = tf_score * idf_score
    # print(query_vector)

    # v1 = np.array([vec1.get(token, 0) for token in term_dic.keys()])
    # v2 = np.array([vec2.get(token, 0) for token in term_dic.keys()])
    # v  = np.array([query_vector.get(token, 0) for token in term_dic.keys()])
    # dot_product1 = np.dot(v1, v)
    # dot_product2 = np.dot(v2, v)
    # print(dot_product1)
    # print(dot_product2)
    # norm_v1 = np.linalg.norm(v1)
    # norm_v2 = np.linalg.norm(v2)
    # norm_v  = np.linalg.norm(v )
    # print(norm_v1)
    # print(norm_v2)
    # print(dot_product1 / (norm_v1 * norm_v))
    # print(dot_product2 / (norm_v2 * norm_v))

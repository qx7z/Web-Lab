import os
from Posting_List.PostingListWithPositionalIndex import load_term_dic,load_filename_to_ID,retrieve_postings_list_only

if __name__ == "__main__":
    postinglist_dir = os.path.join(os.getcwd(),"Lab1" ,"Posting_List")
    index_file = "index_original.bin"
    filename_to_ID_file = "filename_to_ID.bin"
    term_dic = load_term_dic(os.path.join(postinglist_dir,index_file))
    filename_to_ID = load_filename_to_ID(os.path.join(postinglist_dir,filename_to_ID_file))
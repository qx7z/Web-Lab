import os
from XML_Parsing_And_Normalization.XML_Parser import xml_parser
from XML_Parsing_And_Normalization.Tokenizer import tokenize
from XML_Parsing_And_Normalization.Normalize import normalize

if __name__ == "__main__":
    
    Task_2_dir = "XML_Parsing_And_Normalization"
    # 2 XML_Parsing_And_Normalization
    # ---------------------------------------------------------------------------
    # 2.1 Parse xml files
    xml_dir_name = "All_Unpack"
    xml_dir = os.path.join(os.getcwd(),xml_dir_name)
    target_xml_file_pattern = "PastEvent [0-9a-z]*.xml"
    parsed_xml_output_dir_name = "Parsed_xml_files"
    parsed_xml_output_dir = os.path.join(os.getcwd(),Task_2_dir,parsed_xml_output_dir_name)
    os.makedirs(parsed_xml_output_dir, exist_ok=True)

    print(f"\n--- 正在处理目录 '{xml_dir}' 中匹配模式 '{target_xml_file_pattern}' 的文件 ---")
    xml_parser(xml_dir, target_xml_file_pattern,parsed_xml_output_dir)

    # 2.2 tokenize
    parsed_txt_dir = parsed_xml_output_dir
    tokenized_txt_output_dir_name = "Tokenized_txt_files"
    tokenized_txt_output_dir = os.path.join(os.getcwd(),Task_2_dir, tokenized_txt_output_dir_name)
    os.makedirs(tokenized_txt_output_dir, exist_ok=True)
    tokenize(parsed_txt_dir ,tokenized_txt_output_dir)

    # 2.3 Normalization
    tokenized_txt_dir = tokenized_txt_output_dir
    normalized_txt_output_dir_name = "Normalized_txt_files"
    normalized_txt_output_dir = os.path.join(os.getcwd(),Task_2_dir, normalized_txt_output_dir_name)
    os.makedirs(normalized_txt_output_dir, exist_ok=True)
    normalize(tokenized_txt_dir ,normalized_txt_output_dir)
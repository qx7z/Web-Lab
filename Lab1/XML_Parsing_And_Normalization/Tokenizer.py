import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import os
import glob

def tokenize(parsed_txt_dir ,tokenized_txt_output_dir):

    full_pattern = os.path.join(parsed_txt_dir,"*")

    parsed_txt_files = glob.glob(full_pattern)

    print(f"找到 {len(parsed_txt_files)} 个txt文件进行处理。")

    print("\n--- 开始英文分词处理 ---")
    count_chars = 0
    count_files = 0

    for parsed_txt_file in parsed_txt_files:
        with open(parsed_txt_file, 'r', encoding='utf-8') as f:
            descriptions = f.read()
        tokens = word_tokenize(descriptions)
        base = os.path.basename(parsed_txt_file)
        print(f"正在分词: {base}"+ f"词项数: {len(tokens)}")
        output_file_full_path = os.path.join(tokenized_txt_output_dir, base)
        count_files += 1 
        if tokens:
            with open(output_file_full_path, "w", encoding="utf-8") as f:
                content = " ".join(tokens)
                f.write(content)
                count_chars += len(content)
        else:
            print("\n没有内容可写入文件。")

    print(f"\n已将处理后的数据成功写入文件夹: {tokenized_txt_output_dir}，一共{count_files}个文件，一共{count_chars}个字符")

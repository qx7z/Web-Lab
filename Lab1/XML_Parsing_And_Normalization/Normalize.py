import glob
import os
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

nltk.download('omw-1.4')


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    

# 定义一个函数，将 NLTK 的 POS Tag 映射到 WordNetLemmatizer 所需的 POS Tag
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # 默认返回名词

def is_multi_word_phrase(token_str):
    return '_' in token_str

def normalize(tokenized_txt_dir ,normalized_txt_output_dir):
    
    full_pattern = os.path.join(tokenized_txt_dir, "*")

    tokenized_txt_files = glob.glob(full_pattern)

    print(f"找到 {len(tokenized_txt_files)} 个txt文件进行处理。")
    
    print("\n--- 开始文档规范化处理 ---")
    count_lowercasing_chars = 0
    count_without_stopwords_chars = 0
    count_tokens_clean_chars = 0
    count_tokens_lemmatized_chars = 0

    for tokenized_txt_file in tokenized_txt_files:
        tokens = []
        with open(tokenized_txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            tokens = content.strip().split()
        base = os.path.basename(tokenized_txt_file)
        print(f"正在规范化: {base}"+ f"词项数: {len(tokens)}")

        # lowercasing
        lowercasing_tokens = [token.lower() for token in tokens]
        count_lowercasing_chars += len(lowercasing_tokens)
        print(f"\n[1/4] 转换为小写完成 ({len(lowercasing_tokens)} 词项)")

        # 停用词处理
        stop_words = set(stopwords.words('english'))
        tokens_without_stopwords = [
        token for token in lowercasing_tokens if token not in stop_words
        ]
        count_without_stopwords_chars += len(tokens_without_stopwords)
        print(f"[2/4] 去除停用词完成 ({len(tokens_without_stopwords)} 词项)")

        # Punctuation and Number Removal
        punctuation = string.punctuation
        tokens_clean = []
        for token in tokens_without_stopwords:
            if re.fullmatch(r'[-+]?\d+(\.\d+)?', token):
                continue 

            if all(char in punctuation for char in token):
                continue 
    
            tokens_clean.append(token)
        count_tokens_clean_chars += len(tokens_clean)
        print(f"[3/4] 去除数字和纯标点完成 ({len(tokens_clean)} 词项)")

        # stemming
        lemmatizer = WordNetLemmatizer()
        tokens_lemmatized = []
        pos_tagged_tokens = nltk.pos_tag(tokens_clean)
        for token, tag in pos_tagged_tokens:
            if is_multi_word_phrase(token):
                tokens_lemmatized.append(token) # 短语不进行词形还原
            else:
                wordnet_pos = get_wordnet_pos(tag)
                lemmatized_word = lemmatizer.lemmatize(token, pos=wordnet_pos)
                tokens_lemmatized.append(lemmatized_word)
        count_tokens_lemmatized_chars += len(tokens_lemmatized)
        print(f"[4/4] 词形还原完成 ({len(tokens_lemmatized)} 词项)")

        # print(tokens_lemmatized)
        normalized_output_file_path = os.path.join(normalized_txt_output_dir,base)
        with open(normalized_output_file_path, "w", encoding="utf-8") as f:
            f.write(" ".join(tokens_lemmatized)) # 将词语用空格连接后写入文件
            print(f"规范化结果已保存到文件: {normalized_output_file_path}")
    
    print(f"\n[1/4] 转换为小写完成 ({count_lowercasing_chars} 词项)")
    print(f"[2/4] 去除停用词完成 ({count_without_stopwords_chars} 词项)")
    print(f"[3/4] 去除数字和纯标点完成 ({count_tokens_clean_chars} 词项)")
    print(f"[4/4] 词形还原完成 ({count_tokens_lemmatized_chars} 词项)")



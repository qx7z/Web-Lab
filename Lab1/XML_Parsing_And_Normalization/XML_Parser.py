from lxml import etree
import os
import glob
import html
import re

def change_extension(filename, new_extension):
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    if new_extension and not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    return name + new_extension

def extract_event_descriptions_from_xml(xml_file_path):

    descriptions = []
    tree = etree.parse(xml_file_path)
    root = tree.getroot()
    description_elements = root.findall('description')
    # description_element = root.find('description')
    for description_element in description_elements:
        if description_element is not None and description_element.text:
            encoded_text = description_element.text.strip()
            
            # 1. HTML实体解码
            decoded_text = html.unescape(encoded_text)
            
            # 2. 去除HTML标签 (例如 <b>, <br />, <br />)
            # 这正则表达式 <[^>]+> 匹配所有 < 和 > 之间的内容，包括标签本身
            clean_text = re.sub(r'<[^>]+>', '', decoded_text)
            
            # 3. 清理多余的空白字符和换行符
            # 将多个连续的空白字符（包括空格、制表符、换行符）替换为单个空格
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            # print(clean_text)
            descriptions.append(clean_text)
    
    return descriptions

def xml_parser(directory_path, file_pattern, parsed_xml_output_dir):

    full_pattern = os.path.join(directory_path, file_pattern)
    
    # print(full_pattern)

    # glob.glob() 函数返回所有匹配指定模式的路径名列表
    xml_files = glob.glob(full_pattern)

    print(f"找到 {len(xml_files)} 个XML文件进行处理。")

    count_chars = 0
    count_files = 0
    # count_more_than_one_description = 0

    for xml_file in xml_files:
        print(f"正在处理文件: {xml_file}")
        descriptions = extract_event_descriptions_from_xml(xml_file)
        # print(xml_file)
        output_filename = change_extension(xml_file, "txt")
        output_file_full_path = os.path.join(parsed_xml_output_dir, output_filename)
        count_files += 1 
        if descriptions:
            with open(output_file_full_path, "w", encoding="utf-8") as f:
                # if len(descriptions) > 1:
                #     count_more_than_one_description += 1
                content = "\n".join(descriptions)
                f.write(content)
                count_chars += len(content)
        else:
            print("\n没有内容可写入文件。")

    print(f"\n已将处理后的数据成功写入文件夹: {parsed_xml_output_dir}，一共{count_files}个文件，一共{count_chars}个字符")
    # print(f"\n一共有{count_more_than_one_description}个mxl有多个descriptions")

    














    # print("\n--- 合并后的文档 (部分内容) ---")
    # if combined_document:
    #     print(f"{combined_document[:2000]}...") # 打印更多内容以查看多个描述
    # else:
    #     print("没有提取到任何内容。")

    # 清理：删除示例文件和目录（可选）
    # print(f"\n正在清理目录: {output_dir}")
    # if os.path.exists(output_dir):
    #     shutil.rmtree(output_dir)
    #     print(f"已删除目录及其内容: {output_dir}")


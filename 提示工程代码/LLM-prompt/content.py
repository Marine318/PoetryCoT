import os
import json
from glob import glob

def extract_content_between_keywords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 从文章末尾开始寻找关键词
        title_index = None
        rhyme_index = None
        
        # 从后向前遍历
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i].strip()
            if "标题" in line and "《" in line and "》" in line and title_index is None and rhyme_index is not None:
                title_index = i
            if "韵脚" in line and rhyme_index is None:
                rhyme_index = i
        
        # 确保找到两个关键词且顺序正确
        if title_index is not None and rhyme_index is not None and title_index < rhyme_index:
            # 提取从title_index到rhyme_index的内容（包括title_index行，不包括rhyme_index行）
            return ''.join(lines[title_index:rhyme_index])
        
        return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None
    

def create_training_dataset(txt_dir, output_file):
    """
    从txt文件目录创建训练数据集
    
    参数:
        txt_dir: txt文件所在的目录
        output_file: 输出的JSON文件路径
    """
    # 获取所有txt文件路径
    txt_files = glob(os.path.join(txt_dir, "*.txt"))
    dataset = []
    
    for txt_path in txt_files:
        # 读取txt文件内容
        
        content = extract_content_between_keywords(txt_path)   
        
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        
        # 创建对应的图片文件名
        image_file = f"Train/{base_name}.jpg" 
        
        # 构建消息结构
        messages = [
            {
                "role": "system",
                "content": "你是一位精通中国古典诗歌的AI助手,擅长从图像中提炼意境并创作七言绝句。请根据用户提供的图像内容，创作一首符合规范的中文七言绝句。"
            },
            {
                "role": "user",
                "content": "请仔细观察这张图片，结合图像内容创作一首符合图像内容的中文七言绝句。"
            },
            {
                "role": "assistant",
                "content": content
            }
        ]
        
        # 添加到数据集
        dataset.append({
            "messages": messages,
            "images": [image_file]
        })
    
    # 保存为JSON文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"成功创建数据集，包含 {len(dataset)} 个样本，已保存至 {output_file}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    TXT_DIRECTORY = "output_glm"  # 替换为你的txt文件目录
    OUTPUT_JSON = "poetry.json"      # 输出文件名
    
    create_training_dataset(TXT_DIRECTORY, OUTPUT_JSON)
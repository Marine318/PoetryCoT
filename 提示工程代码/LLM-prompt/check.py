def check_poem_structure(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    required_keywords = ["标题", "起", "承", "转", "合"]

    for i, line in enumerate(lines):
        if '"assistant"' in line:
            next_index = i + 1
            if next_index < len(lines):
                next_line = lines[next_index]
                # 检查是否包含所有关键词
                if not all(keyword in next_line for keyword in required_keywords):
                    output_index = i + 5
                    if output_index < len(lines):
                        print(f"❌ 缺失关键词，位置: 行 {i+1}，输出第 {output_index+1} 行内容：")
                        print(lines[output_index].strip())
                        print('-' * 60)

check_poem_structure("poetry.json")
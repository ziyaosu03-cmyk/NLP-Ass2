#!/usr/bin/env python3
import json
import re

# 读取notebook文件
with open('assignment_2.ipynb', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到问题所在：第168行之后有重复的代码
# 第168行是:        "\"\"
# 第169行开始是重复的代码片段，应该删除

# 找到Task 1代码cell的结束位置
# 查找包含"\"\""的行，这应该是docstring的结束
problem_start = None
for i, line in enumerate(lines):
    if i >= 160 and i < 170:
        if '""' in line and '"""' not in line:
            # 这可能是docstring结束，但后面有重复代码
            if i == 167:  # 第168行（0-indexed是167）
                problem_start = i + 1
                break

# 查找重复代码的结束位置（应该是下一个cell的开始）
if problem_start:
    # 查找下一个cell的开始标记
    end_pos = None
    for i in range(problem_start, min(problem_start + 30, len(lines))):
        if 'cell_type' in lines[i] or (i > problem_start and '],' in lines[i] and 'source' not in lines[i-5:i]):
            # 检查是否是cell的结束
            if i > problem_start + 10:  # 至少跳过几行
                end_pos = i
                break
    
    if end_pos:
        print(f"找到重复代码: 行 {problem_start+1} 到 {end_pos}")
        # 删除重复的代码
        lines = lines[:problem_start] + lines[end_pos:]
        print(f"已删除重复代码")
    else:
        # 如果找不到明确的结束位置，手动删除到特定行
        # 查找包含"# print(f\\\"Perplexity"的最后一行
        for i in range(problem_start, min(problem_start + 20, len(lines))):
            if 'Perplexity' in lines[i] and 'print' in lines[i]:
                end_pos = i + 1
                break
        if end_pos:
            lines = lines[:problem_start] + lines[end_pos:]
            print(f"已删除重复代码到行 {end_pos}")

# 将修改后的内容写回文件
with open('assignment_2.ipynb', 'w', encoding='utf-8') as f:
    f.writelines(lines)

# 现在尝试解析JSON
try:
    with open('assignment_2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("JSON格式已修复")
    
    # 现在更新Task 3
    if len(nb['cells']) > 3:
        task3_cell = nb['cells'][3]
        
        if task3_cell['cell_type'] == 'code':
            new_code_lines = [
                "# Your code for Task 3 here",
                "from sklearn.feature_extraction.text import TfidfVectorizer",
                "from sklearn.metrics.pairwise import cosine_similarity",
                "import numpy as np",
                "",
                "document_corpus = [",
                "    \"The field of machine learning has seen rapid growth in recent years, especially in deep learning.\",",
                "    \"Natural language processing allows machines to understand and respond to human text.\",",
                "    \"Computer vision focuses on enabling computers to see and interpret the visual world.\",",
                "    \"Deep learning models like convolutional neural networks are powerful for computer vision tasks.\",",
                "    \"Recurrent neural networks are often used for sequential data in natural language processing.\",",
                "    \"The advances in reinforcement learning have led to breakthroughs in game playing and robotics.\",",
                "    \"Transfer learning enables models trained on large datasets to be adapted for new tasks with limited data.\",",
                "    \"Unsupervised learning techniques can discover hidden patterns in data without labeled examples.\",",
                "    \"Optimization algorithms such as stochastic gradient descent are crucial for training neural networks.\",",
                "    \"Attention mechanisms have improved the performance of natural language translation and image captioning.\",",
                "    \"Generative adversarial networks create realistic images and are used for data augmentation.\",",
                "    \"Feature engineering and selection are important steps in classical machine learning pipelines.\",",
                "    \"Object detection is a key task in computer vision that involves locating instances within images.\",",
                "    \"The combination of convolutional and recurrent networks is used for video classification tasks.\",",
                "    \"Zero-shot learning allows models to recognize objects and concepts they have not seen during training.\",",
                "    \"Natural language generation is used for creating text summaries and chatbot responses.\",",
                "    \"Graph neural networks leverage graph structures for tasks such as social network analysis and chemistry.\",",
                "    \"Hyperparameter tuning can significantly improve the accuracy of deep learning models.\",",
                "    \"Cross-modal learning involves integrating information from multiple data sources such as text and images.\",",
                "    \"Evaluating model performance requires a good choice of metrics such as F1-score and RMSE.\"",
                "]",
                "",
                "# Step 2: Create a TfidfVectorizer and fit it on the corpus",
                "vectorizer = TfidfVectorizer()",
                "vectorizer.fit(document_corpus)",
                "",
                "# Step 3: Transform the corpus into a TF-IDF document-term matrix",
                "doc_term_matrix = vectorizer.transform(document_corpus)",
                "",
                "def rank_documents(query, vectorizer, doc_term_matrix, top_n=3):",
                "    \"\"\"",
                "    根据查询对文档进行排序",
                "    ",
                "    参数:",
                "        query: 查询字符串",
                "        vectorizer: 已拟合的TfidfVectorizer",
                "        doc_term_matrix: 文档-词TF-IDF矩阵",
                "        top_n: 返回前N个最相关的文档（默认3）",
                "    ",
                "    返回:",
                "        包含(top_n个文档的索引, 文档内容, 相似度分数)的列表，按相似度降序排列",
                "    \"\"\"",
                "    # 将查询转换为TF-IDF向量（使用相同的vectorizer）",
                "    query_vector = vectorizer.transform([query])",
                "    ",
                "    # 计算查询向量与所有文档向量之间的余弦相似度",
                "    similarities = cosine_similarity(query_vector, doc_term_matrix).flatten()",
                "    ",
                "    # 获取前top_n个最相似文档的索引（降序排列）",
                "    top_indices = np.argsort(similarities)[::-1][:top_n]",
                "    ",
                "    # 返回索引和对应的文档内容",
                "    ranked_docs = [(idx, document_corpus[idx], similarities[idx]) for idx in top_indices]",
                "    ",
                "    return ranked_docs",
                "",
                "# Step 5: Demonstrate the system with the query",
                "query = \"deep learning models for vision\"",
                "ranked_docs = rank_documents(query, vectorizer, doc_term_matrix, top_n=3)",
                "",
                "print(f\"Top {len(ranked_docs)} documents for the query: '{query}'\\n\")",
                "for rank, (idx, doc, similarity) in enumerate(ranked_docs, 1):",
                "    print(f\"Rank {rank} (Similarity: {similarity:.4f}):\")",
                "    print(f\"  Document {idx}: {doc}\\n\")"
            ]
            
            task3_cell['source'] = new_code_lines
            
            with open('assignment_2.ipynb', 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            
            print('Task 3代码已成功更新')
        else:
            print(f'Cell 3不是代码cell')
    else:
        print('Notebook中没有足够的cells')
        
except json.JSONDecodeError as e:
    print(f"JSON仍有错误: {e}")
    print("需要手动修复JSON格式")

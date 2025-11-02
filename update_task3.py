#!/usr/bin/env python3
import json
import re

# 读取notebook文件
with open('assignment_2.ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# 尝试解析JSON
try:
    nb = json.loads(content)
except json.JSONDecodeError as e:
    print(f"JSON解析错误: {e}")
    print("尝试修复JSON格式...")
    # 如果JSON有格式问题，可能需要特殊处理
    # 先尝试用正则表达式找到Task 3的代码部分
    exit(1)

# 找到Task 3的代码cell（索引3）
if len(nb['cells']) > 3:
    task3_cell = nb['cells'][3]
    
    # 确保这是代码cell
    if task3_cell['cell_type'] == 'code':
        # 替换代码内容
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
        
        # 更新cell的source
        task3_cell['source'] = new_code_lines
        
        # 保存notebook文件
        with open('assignment_2.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print('Task 3代码已成功更新')
    else:
        print(f'Cell 3不是代码cell，而是: {task3_cell["cell_type"]}')
else:
    print('Notebook中没有足够的cells')

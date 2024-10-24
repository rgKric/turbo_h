from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import os
import json

# я хз нахера это, но без этого не работает(((((((
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Функция для получения эмбеддинга текста
def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Допустим, у нас есть список абзацев или фрагментов текста документов
def create_index(corpus, tokenizer, model):
    # Получаем эмбеддинги для всех фрагментов
    embeddings = np.array([get_embedding(text, tokenizer, model) for text in corpus])
    # Создаём индекс FAISS
    dimension = embeddings.shape[1]
    print(dimension)
    index = faiss.IndexFlatL2(dimension)  # Используем L2-норму для поиска
    index.add(embeddings)  # Добавляем эмбеддинги в индекс
    return index

def ranking(question, index, tokenizer, model, k):
    # Получаем эмбеддинг для вопроса
    question_embedding = get_embedding(question, tokenizer, model).reshape(1, -1)
    # Поиск ближайших фрагментов
    # k - Количество ближайших соседей
    _, indices = index.search(question_embedding, k)
    return indices

def global_ranking(question, file, tokenizer, model, k=3):
    with open(file, 'r', encoding='utf-8') as f:
        json_file = json.load(f)['data']
    # условно берем максимум 3 первых выпавших, игноря с steps
    content_corpus = []
    for i in range(len(json_file)):
        if list(json_file[i].keys())[1] != 'steps':
            content_corpus += json_file[i]['content']
    # теперь выставляем ранги корпусам
    content_index = create_index(content_corpus, tokenizer, model)
    content_indices = ranking(question, content_index, tokenizer, model, k)[0]
    return content_corpus, content_indices


# Выводим найденные фрагменты
if __name__ == '__main__':
    path = 'docs/docs_json/Service_registratsii.json'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    question = 'Как заполнять телефон?'

    content_corpus, content_idxs = global_ranking(question, path, tokenizer, model, k=5)
    for idx in content_idxs:
        print(content_corpus[idx])

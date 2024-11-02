import json
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
import nltk
import re
import pymorphy3


def split_document(json_file, tokenizer, max_context_length=450, overlap=100):
    '''
    Split every title to chunks with len=max_context_length with crossing=overlap

    :param json_file: json файл
    :param tokenizer: токенайзер (тут T5)
    :param max_context_length: максимальная длина чанка (в токенах)
    :param overlap: длина перекрывания (тоже в токенах)
    :return: Список из title + content, который поделен на чанки
    '''
    # читаем файл, получаем список
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    # разделяем на пересекающиеся чанки каждый блок (по токенам)
    for i in range(len(data)):
        # добавляем заголовок в текст, мб поможет
        title = data[i]['title']
        content = data[i]['content']
        if title not in content:
            content = title + '. ' + content
        content_tokens = tokenizer.encode(content)
        # бьем на куски если длина текста меньше максимальной допустимой длины
        if len(content_tokens) < max_context_length:
            data[i]['content'] = [content]
        else:
            new_content = []
            step = max_context_length - overlap
            for j in range(0, len(content_tokens) - overlap, step):
                chunk = content_tokens[j:j + max_context_length]
                text = ''.join(tokenizer.batch_decode(chunk, skip_special_tokens=True))
                new_content.append(text)
            data[i]['content'] = new_content
    return data


def cos_similarities(embeddings, q_embedding):
    similarities = []
    for i in range(len(embeddings)):
        similarity = cosine_similarity(q_embedding, embeddings[i]).mean()
        similarities.append(similarity)
    return similarities


def euclidean_similarities(embeddings, q_embedding):
    similarities = []
    for i in range(len(embeddings)):
        similarity = euclidean_distances(q_embedding, embeddings[i]).mean()
        similarities.append(similarity)
    return similarities


def cleaning(content, ru_stops, n):
    morph = pymorphy3.MorphAnalyzer()
    text = content.lower()
    text = re.sub(r'[^\w\s]', '', text)

    words = text.split()
    processed_words = []
    for word in words:
        if word not in ru_stops:
            lemma = morph.parse(word)[0].normal_form
            processed_words.append(lemma)
    n_grams = set()
    for i in range(1, n + 1):
        i_gram = set(nltk.ngrams(processed_words, i))
        n_grams = n_grams.union(i_gram)
    return set(n_grams)

def precision(context, question, n=3):
    # нормализация нашего чудесного текста
    with open('stops.txt', 'r', encoding='utf-8') as f:
        ru_stops = set(f.read().splitlines())
    set_question = cleaning(question, ru_stops, n)
    set_context = cleaning(context, ru_stops, n)
    score = len(set_context.intersection(set_question))/len(set_question)
    return score


def get_embeddings(text, tokenizer, model):
    model.eval()
    tokens = tokenizer.encode_plus(text, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=tokens['input_ids'], decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True)

    last_hidden_state = outputs.encoder_last_hidden_state
    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
    return embedding


def relevant_chunk_finder(data_list,
                          question,
                          tokenizer,
                          model,
                          similarity_type='precision'):
    '''
    Выбирает наиболее релевантный раздел, основываясь на метрике similarity_type

    :param data_list: List[dict{title, content, ...}] - список заголовков с контекстами и их векторными представлениями
    :param question: вопрос в формате str
    :param tokenizer: tokenizer
    :param model: model
    :param similarity_type: способ сравнения схожести вектора вопроса и контекста ['precision', 'cosin', 'euclidean']
    :return: раздел, который ближе всего к вопросу в формате List[str]
    '''
    model.eval()
    # если нет эмбеддингов, то создаем их
    if 'embeddings' in list(data_list[0].keys()):
        data = data_list.copy()
    else:
        print('no embeddings, generating...')
        data = data_list.copy()
        for i in tqdm(range(len(data))):
            content = data[i]['content']
            data[i]['embeddings'] = []
            for text in content:
                embedding = get_embeddings(text, tokenizer, model)
                data[i]['embeddings'].append(embedding)

    # если precision, то схожесть по множеству слов, иначе схожесть эмбеддингов
    if similarity_type == 'precision':
        # разные n-грамы, потому что если есть сходство для 3х, то это лучше, чем по 1, кмк
        similarities = []
        for i in range(len(data)):
            max_similarity = 0
            for j in range(len(data[i]['content'])):
                content = data[i]['content'][j]
                similarity = precision(context=content, question=question)
                if max_similarity < similarity:
                    max_similarity = similarity
            similarities.append(max_similarity)
    else:
        # получаем эмбеддинг вопроса
        print('Question embedding generating!')
        q_embedding = get_embeddings(question.replace('?', ''), tokenizer, model)
        print('DONE')
        embeddings = [data[i]['embeddings'] for i in range(len(data))]
        # находим среднее косинусное расстояние для каждого раздела до вопроса
        if similarity_type == 'cosin':
            similarities = cos_similarities(embeddings, q_embedding)
        elif similarity_type == 'euclidean':
            similarities = euclidean_similarities(embeddings, q_embedding)

    # берем самые крутые тексты (мб и 1 текст)
    most_relevant = np.argmax(similarities)
    return data[most_relevant]['content'], similarities


def get_answer(contexts, question, tokenizer, model):
    model.eval()
    QA_PROMPT = "Сгенерируй ответ на вопрос по тексту. Текст: '{context}'. Вопрос: '{question}'."
    confidence_scores = []
    answers_tokens = []
    print('Start finding!')
    for context in contexts:
        prompt = QA_PROMPT.format(context=context, question=question)
        prompt_tokens = tokenizer.encode_plus(prompt, return_tensors='pt')
        prompt_tokens = {k: v.to(model.device) for k, v in prompt_tokens.items()}

        #получаем ответ и все логиты
        with torch.no_grad():
            outputs = model.generate(**prompt_tokens,
                                     max_new_tokens=64,
                                     do_sample=True,
                                     num_beams=3,
                                     num_return_sequences=1,
                                     temperature=0.8,
                                     top_p=0.9,
                                     top_k=50,
                                     output_scores=True,
                                     return_dict_in_generate=True)

        # Получаем токены и лог-вероятности
        generated_tokens = outputs.sequences[0]
        logits = outputs.scores
        probabilities = [torch.softmax(logit, dim=-1) for logit in logits]
        predicted_probabilities = []
        for i, token_id in enumerate(generated_tokens[1:]):
            predicted_probabilities.append(max(probabilities[i][:, token_id]).item())

        # Уверенность ответа
        confidence_score = sum(predicted_probabilities) / len(predicted_probabilities)
        confidence_scores.append(confidence_score)
        answers_tokens.append(generated_tokens)
    print('DONE')
    best_idx = np.argmax(confidence_scores)
    print(contexts[best_idx])
    print(np.max(confidence_scores))
    answer = tokenizer.decode(answers_tokens[best_idx], skip_special_tokens=True)
    return answer

# сначала relevant_chunk_finder для всех текстов в файле
# потом get_answer
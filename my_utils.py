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
        similarity = cosine_similarity(q_embedding, embeddings[i]).max()
        similarities.append(similarity)
    return similarities


def euclidean_similarities(embeddings, q_embedding):
    similarities = []
    for i in range(len(embeddings)):
        similarity = euclidean_distances(q_embedding, embeddings[i]).max()
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

    n_grams = set(nltk.ngrams(processed_words, n))
    return n_grams

def precision(context, question, n=2):
    # нормализация нашего чудесного текста
    with open('stops.txt', 'r', encoding='utf-8') as f:
        ru_stops = set(f.read().splitlines())
    set_question = cleaning(question, ru_stops, n)
    set_context = cleaning(context, ru_stops, n)
    score = len(set_context.intersection(set_question))/len(set_question)
    return score


def calculate_avg_rank(rankings_A, rankings_B, weight_A, weight_B, n=4):
    rankings_dict = {}
    for idx, arg in enumerate(rankings_A):
        if arg not in rankings_dict:
            rankings_dict[arg] = []
        rankings_dict[arg].append(idx + 1)

    for idx, arg in enumerate(rankings_B):
        if arg not in rankings_dict:
            rankings_dict[arg] = []
        rankings_dict[arg].append(idx + 1)
    # Calculate average rank for each argument
    avg_rank_dict = {arg: ranks[0]*weight_A + ranks[1]*weight_B for arg, ranks in rankings_dict.items()}
    # Sort the arguments based on their average rank in ascending order (lower rank is better)
    sorted_avg_rankings = sorted(avg_rank_dict.items(), key=lambda x: x[1])
    best = [sorted_avg_rankings[i][0] for i in range(n)]
    return best


def get_embeddings(text, tokenizer, model):
    model.eval()
    tokens = tokenizer.encode_plus(text, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=tokens['input_ids'], decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True)

    last_hidden_state = outputs.encoder_last_hidden_state
    embedding = last_hidden_state.mean(dim=1).cpu().tolist()
    return embedding


def relevant_chunk_finder(data_list,
                          question,
                          tokenizer,
                          model, n=4):
    '''
    Выбирает наиболее релевантный раздел, основываясь на метрике similarity_type

    :param data_list: List[dict{title, content, ...}] - список заголовков с контекстами и их векторными представлениями
    :param question: вопрос в формате str
    :param tokenizer: tokenizer
    :param model: model
    :param n: колчество релевантных контекстов
    :return: раздел, который ближе всего к вопросу в формате List[str]
    '''
    model.eval()
    # если нет эмбеддингов, то создаем их
    if 'embeddings' in list(data_list[0].keys()):
        data = data_list.copy()
    else:
        data = data_list.copy()
        for i in tqdm(range(len(data))):
            content = data[i]['content']
            data[i]['embeddings'] = []
            for text in content:
                embedding = get_embeddings(text, tokenizer, model)
                data[i]['embeddings'].append(embedding)

    # расчет precision
    similarities_prec = []
    for i in range(len(data)):
        max_similarity = 0
        for j in range(len(data[i]['content'])):
            content = data[i]['content'][j]
            similarity = precision(context=content, question=question)
            if max_similarity < similarity:
                max_similarity = similarity
        similarities_prec.append(max_similarity)
    # получаем эмбеддинг вопроса
    q_embedding = get_embeddings(question.replace('?', ''), tokenizer, model)
    embeddings = [data[i]['embeddings'] for i in range(len(data))]
    # находим среднее косинусное сходство для каждого раздела до вопроса
    similarities_cosin = cos_similarities(embeddings, q_embedding)

    # если в precision не все 0, то ранжируем, иначе только косинусное сходство
    if sum(similarities_prec) != 0:
        best = calculate_avg_rank(np.argsort(similarities_prec)[::-1],
                                  np.argsort(similarities_cosin)[::-1],
                                  0.7, 0.3, n=n)
    else:
        best = np.argsort(similarities_cosin)[::-1][:n]
    most_relevant = []
    for i in best:
        most_relevant += data[i]['content']
    return most_relevant, best


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
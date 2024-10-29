import json
import torch
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

def relevant_chunk_finder(data_list, question, tokenizer, model, embedding_type='last'):
    # если нет эмбеддингов, то создаем их
    if 'embeddings' not in list(data_list[0].keys()):
        data = data_list.copy()
        model.eval()
        for i in tqdm(range(len(data))):
            content = data[i]['content']
            data[i]['embeddings'] = []
            for text in content:
                tokens = tokenizer.encode_plus(text, return_tensors='pt')
                tokens = {k: v.to(model.device) for k, v in tokens.items()}
                decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

                if embedding_type == 'last':
                    with torch.no_grad():
                        outputs = model(input_ids=tokens['input_ids'], decoder_input_ids=decoder_input_ids,
                                        output_hidden_states=True)

                    last_hidden_state = outputs.encoder_last_hidden_state
                    embedding = last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
                    data[i]['embeddings'].append(embedding)
    else:
        data = data_list

    # получаем эмбеддинг вопроса
    q_tokens = tokenizer.encode_plus(question.replace('?', ''), return_tensors='pt')
    q_tokens = {k: v.to(model.device) for k, v in q_tokens.items()}
    decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]]).to(device)

    if embedding_type == 'last':
        with torch.no_grad():
            outputs = model(input_ids=q_tokens['input_ids'], decoder_input_ids=decoder_input_ids,
                            output_hidden_states=True)

        last_hidden_state = outputs.encoder_last_hidden_state
        q_embedding = last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

    # находим среднее косинусное расстояние для каждого раздела до вопроса
    similarities = []
    for i in range(len(data)):
        embeddings = data[i]['embeddings']
        similarity = cosine_similarity(q_embedding, embeddings).mean()
        similarities.append(similarity)

    # берем самые крутые тексты (мб и 1 текст)
    most_relevant = np.argmax(similarities)
    return data[most_relevant]['content']

def get_answer(contexts, question, tokenizer, model):
    QA_PROMPT = "Сгенерируй ответ на вопрос по тексту. Текст: '{context}'. Вопрос: '{question}'."
    confidence_scores = []
    answers_tokens = []
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
                                     return_dict_in_generate=True,
                                     early_stopping=True)

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

    best_idx = np.argmax(confidence_scores)
    answer = tokenizer.decode(answers_tokens[best_idx], skip_special_tokens=True)
    return answer

# сначала relevant_chunk_finder для всех текстов в файле
# потом get_answer

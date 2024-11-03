from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
from my_utils import *
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

# Словарь с вариантами файлов для кнопок
file_options = {
    "file1": "Naumen Desk",
    "file2": "Service registratsii",
}

# Хранилище данных пользователя
user_data = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Доброго времени суток! Здесь вы можете задать вопросы по разным документациям, и, возможно, я смогу на них ответить, но ничего не гарантирую.\n Для того, чтобы задать новый вопрос, введите команду /new_question'
    )


# Команда /new_question для нового вопроса
async def new_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton(name, callback_data=key)] for key, name in file_options.items()]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        "Выберите вариант файла:",
        reply_markup=reply_markup
    )


# Обработка выбора кнопки
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    # Сохранить выбранный файл
    selected_option = query.data
    user_data[query.from_user.id] = {"file": file_options[selected_option]}

    await query.edit_message_text(text=f"Вы выбрали: {file_options[selected_option]}. Теперь задайте свой вопрос.")


# Обработка текста (ожидание вопроса от пользователя)
async def question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    # Проверяем, выбрал ли пользователь файл
    if user_id in user_data:
        user_question = update.message.text
        selected_file = user_data[user_id]["file"]

        # Сохраняем вопрос
        user_data[user_id]["question"] = user_question

        # Отправляем подтверждение
        await update.message.reply_text(f"Вы задали вопрос: '{user_question}' для файла '{selected_file}'.\n Это може занять пару минут...")

        # открываем файл, который выбрал пользователь и передаем его с вопросом в finder:
        file_name = selected_file.replace(' ', '_')
        file_name = f'{file_name}_splited_emb.json'
        path_to_file = os.path.join(context.bot_data['path_to_docs_dir'], file_name)
        # если вопрос короткий или вызывает ошибку (берутся n_grams), о просто просим перезадать вопрос нормально
        try:
            with open(path_to_file, 'r', encoding='utf-8') as f:
                data_list = json.load(f)['data']
            relevant_chunks, _ = relevant_chunk_finder(data_list=data_list,
                                                       question=user_question,
                                                       tokenizer=context.bot_data['tokenizer'],
                                                       model=context.bot_data['model'])
            # теперь генерируем ответ
            answer = get_answer(contexts=relevant_chunks,
                                question=user_question,
                                tokenizer=context.bot_data['tokenizer'],
                                model=context.bot_data['model'])
        except:
            answer = user_question
        # Жуткий костыль, мне очень стыдно
        if answer == user_question:
            await update.message.reply_text(text='Пожалуйста, перефразируйте вопрос, так как я не смог найти ответ...')
        else:
            await update.message.reply_text(text=f'Ответ на ваш вопрос:\n{answer}\n')
    else:
        await update.message.reply_text("Сначала выберите вариант файла с помощью команды /new_question")


# Основной запуск бота
if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # загрузка токена
    load_dotenv('token.env')
    TOKEN = os.getenv("TG_TOKEN")
    app = ApplicationBuilder().token(TOKEN).build()

    # загрузка модели и токенизатор и пути к файлам
    if os.path.exists('./t5_model/'):
        saved_checkpoint = './t5_model/'
    else:
        saved_checkpoint = 'hivaze/AAQG-QA-QG-FRED-T5-1.7B'
    tokenizer = AutoTokenizer.from_pretrained(saved_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(saved_checkpoint)
    app.bot_data['model'] = model.to(device)
    app.bot_data['tokenizer'] = tokenizer
    app.bot_data['path_to_docs_dir'] = 'docs/docs_json/'

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new_question", new_question))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler))

    print("Бот запущен...")
    app.run_polling()
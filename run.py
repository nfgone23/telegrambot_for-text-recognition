import os
import asyncio
import logging
import keras
import keyboards as kb
from dotenv import load_dotenv
from emnistmain import letters_extract, img_letters_to_str
from mnistmain import numbers_extract, img_numbers_to_str

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.fsm.storage.memory import MemoryStorage


load_dotenv()
bot = Bot(os.getenv("TOKEN"))
dp = Dispatcher(storage=MemoryStorage())

# хранение состояний
class Form(StatesGroup):
    select_button = State()
    recognize_text = State()
    recognize_numbers = State()



@dp.message(Command("start"))
async def command_start(message: Message, state: FSMContext) -> None:
    # Приветствие, предоставление выбора
    await state.set_state(Form.select_button)
    await message.answer(
        f"Добро пожаловать {message.from_user.first_name}. Меня зовут Ocr бот.\n"
        "Выберите вид распознавания:",
        reply_markup=kb.buttons
    )


# Блок: просьба отправить изображение
@dp.message(Form.select_button, F.text.casefold() == "распознавание текста")
async def recognize_text(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.recognize_text)

    await message.answer(
        "Пришлите изображение для распознавания текста",
        reply_markup=ReplyKeyboardRemove(),
    )
    

@dp.message(Form.select_button, F.text.casefold() == "распознавание чисел")
async def recognize_numbers(message: Message, state: FSMContext) -> None:
    await state.set_state(Form.recognize_numbers)

    await message.answer(
        "Пришлите изображение для распознавания чисел",
        reply_markup=ReplyKeyboardRemove(),
    )


@dp.message(Form.select_button)
async def proccess_unknown_write(message: Message, state: FSMContext) -> None:
    await message.reply("Я не понимаю тебя :(")

# Блок загрузки изображения и распознавания
@dp.message(Form.recognize_text)
async def recognize_photo_text(message: Message, state: FSMContext) -> None:
    # Файл так как телеграм сжимает качество фото)
    await state.clear()
    document = message.document
    await bot.download(document)
    letters_extract("image.png")
    model = keras.models.load_model("emnist_model.keras")
    s_out = img_letters_to_str(model, "image.png")
    await message.answer(
        f"Распознанный текст: {str(s_out)}."
    )
    

@dp.message(Form.recognize_numbers)
async def recognize_photo_numbers(message: Message, state: FSMContext) -> None:
    # Файл так как телеграм сжимает качество фото)
    await state.clear()
    document = message.document
    await bot.download(document)
    numbers_extract("image.png")
    model = keras.models.load_model("mnist_model.keras")
    s_out = img_numbers_to_str(model, "image.png")
    await message.answer(
        f"Распознанные числа: {str(s_out)}."
    )
    

@dp.message(Command("cancel"))
@dp.message(F.text.casefold() == "cancel")
async def cancel(message: Message, state: FSMContext) -> None:
    """Разрешить пользователю отменить любое действие"""
    current_state = await state.get_state()
    if current_state is None:
        return
    
    logging.info("Cancelling state %r", current_state)
    await state.clear()
    await message.answer(
        "Cancelled.",
        reply_markup=ReplyKeyboardRemove(),
    )


async def main():   
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)

main_buttons = [
    [KeyboardButton(text="Распознавание текста"),
     KeyboardButton(text="Распознавание чисел")]
]

buttons = ReplyKeyboardMarkup(keyboard=main_buttons,
                              resize_keyboard=True,
                              input_field_placeholder="Выберите")
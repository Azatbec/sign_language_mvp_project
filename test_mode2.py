"""
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# === ШРИФТЫ ===
try:
    font_progress = ImageFont.truetype("arial.ttf", 24)
    font_text = ImageFont.truetype("arial.ttf", 40)
    font_hint = ImageFont.truetype("arial.ttf", 32)
except:
    font_progress = ImageFont.load_default()
    font_text = ImageFont.load_default()
    font_hint = ImageFont.load_default()

def draw_text_pil(frame, text, x, y, font, color=(0,0,0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)

# === НАСТРОЙКИ ===
GESTURE_FOLDER = "alfavit_parts"

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

TEST_TEXT = "ДОМ НА КРАЮ ОПУШКИ ПРИВЕТ"

SHOW_TIME = 1.8

GESTURE_SIZE = (300, 420)

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

LEFT_AREA_WIDTH = WINDOW_WIDTH // 3
RIGHT_AREA_X = LEFT_AREA_WIDTH

PADDING = 30
IMAGE_AREA_WIDTH = GESTURE_SIZE[0] + 2 * PADDING
IMAGE_AREA_HEIGHT = GESTURE_SIZE[1] + 2 * PADDING
IMAGE_AREA_X = RIGHT_AREA_X + (WINDOW_WIDTH - RIGHT_AREA_X - IMAGE_AREA_WIDTH) // 2
IMAGE_AREA_Y = 50

BOTTOM_BAR_HEIGHT = 80
BOTTOM_BAR_Y = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT - 80

# === ЗАГРУЗКА ===
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, GESTURE_SIZE)

print(f"Загружено жестов: {len(gesture_images)} из {len(CLASSES)}")

letters = [c.upper() for c in TEST_TEXT if c.upper() in gesture_images]

if not letters:
    print("Нет букв для показа.")
    exit()

print(f"\nТекст: \"{TEST_TEXT}\" ({len(letters)} букв)\n")

# === ПОКАЗ ===
i = 0
while True:
    letter = letters[i]

    frame = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

    # Левая область — текст
    left_text = "Область текста\nВ будущем здесь\nбудет ввод текста"
    lines = left_text.split('\n')
    y_start = 100
    for line in lines:
        frame = draw_text_pil(frame, line, 50, y_start, font_text, (100, 100, 100))
        y_start += 60

    # Рамка с изображением
    cv2.rectangle(frame, (IMAGE_AREA_X, IMAGE_AREA_Y),
                  (IMAGE_AREA_X + IMAGE_AREA_WIDTH, IMAGE_AREA_Y + IMAGE_AREA_HEIGHT),
                  (180, 180, 180), 4)

    # Жест внутри рамки
    img = gesture_images[letter]
    h, w = img.shape[:2]
    start_x = IMAGE_AREA_X + PADDING
    start_y = IMAGE_AREA_Y + PADDING

    frame[start_y:start_y + h, start_x:start_x + w] = img

    # Нижняя полоса — прогресс
    cv2.rectangle(frame, (0, BOTTOM_BAR_Y), (WINDOW_WIDTH, WINDOW_HEIGHT), (240, 240, 240), -1)

    progress = (i + 1) / len(letters)
    bar_width = 600
    bar_x = (WINDOW_WIDTH - bar_width) // 2
    bar_y = BOTTOM_BAR_Y + 20

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (220, 220, 220), -1)
    filled = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 30), (70, 180, 70), -1)

    progress_text = f"{i + 1} / {len(letters)}"
    frame = draw_text_pil(frame, progress_text, bar_x + bar_width // 2 - 50, bar_y + 5,
                          font_progress, (0, 0, 0))

    # Подсказки — обновлены под 4 и 6
    hint_y = WINDOW_HEIGHT - 50
    frame = draw_text_pil(frame, "4 — назад    6 — вперёд    ESC / q — выйти", WINDOW_WIDTH // 2 - 300, hint_y,
                          font_hint, (100, 100, 100))

    cv2.imshow("Дактиль РЖЯ — Показ жестов", frame)

    key = cv2.waitKey(int(SHOW_TIME * 1000))

    if key == 27 or key == ord('q') or key == ord('Q'):  # ESC или q
        print("Выход")
        break
    elif key == ord('4'):  # Клавиша 4 — назад
        i = (i - 1) % len(letters)
    elif key == ord('6'):  # Клавиша 6 — вперёд
        i = (i + 1) % len(letters)

print("\nПоказ завершён!")
cv2.destroyAllWindows()
"""
"""
0
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from tkinter import Tk
from tkinter.simpledialog import askstring

# === ШРИФТЫ ===
try:
    font_progress = ImageFont.truetype("arial.ttf", 24)
    font_text = ImageFont.truetype("arial.ttf", 40)
    font_hint = ImageFont.truetype("arial.ttf", 32)
except:
    font_progress = ImageFont.load_default()
    font_text = ImageFont.load_default()
    font_hint = ImageFont.load_default()

def draw_text_pil(frame, text, x, y, font, color=(0,0,0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color)
    return np.array(img_pil)

# === НАСТРОЙКИ ===
GESTURE_FOLDER = "alfavit_parts"

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

SHOW_TIME = 1.8

GESTURE_SIZE = (300, 420)

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

LEFT_AREA_WIDTH = WINDOW_WIDTH // 3
RIGHT_AREA_X = LEFT_AREA_WIDTH

PADDING = 30
IMAGE_AREA_WIDTH = GESTURE_SIZE[0] + 2 * PADDING
IMAGE_AREA_HEIGHT = GESTURE_SIZE[1] + 2 * PADDING
IMAGE_AREA_X = RIGHT_AREA_X + (WINDOW_WIDTH - RIGHT_AREA_X - IMAGE_AREA_WIDTH) // 2
IMAGE_AREA_Y = 50

BOTTOM_BAR_HEIGHT = 80
BOTTOM_BAR_Y = WINDOW_HEIGHT - BOTTOM_BAR_HEIGHT - 80

# === ЗАГРУЗКА ===
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            img_data = f.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, GESTURE_SIZE)

print(f"Загружено жестов: {len(gesture_images)} из {len(CLASSES)}")

# === ВВОД ТЕКСТА ОТ ПОЛЬЗОВАТЕЛЯ ===
Tk().withdraw()  # Скрываем главное окно tkinter
user_text = askstring("Ввод текста", "Введите текст для показа жестов (русские буквы):")
if not user_text:
    print("Текст не введён. Выход.")
    exit()

TEST_TEXT = user_text.upper()  # Приводим к верхнему регистру

letters = [c for c in TEST_TEXT if c in gesture_images]

if not letters:
    print("В введённом тексте нет поддерживаемых букв.")
    exit()

print(f"\nТекст: \"{user_text}\"")
print(f"Будет показано {len(letters)} жестов\n")

# === ПОКАЗ ===
i = 0
while True:
    letter = letters[i]

    frame = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

    # Левая область — введённый текст
    left_lines = user_text.splitlines()
    y_start = 100
    for line in left_lines:
        frame = draw_text_pil(frame, line, 50, y_start, font_text, (100, 100, 100))
        y_start += 60

    # Рамка с изображением
    cv2.rectangle(frame, (IMAGE_AREA_X, IMAGE_AREA_Y),
                  (IMAGE_AREA_X + IMAGE_AREA_WIDTH, IMAGE_AREA_Y + IMAGE_AREA_HEIGHT),
                  (180, 180, 180), 4)

    # Жест внутри рамки
    img = gesture_images[letter]
    h, w = img.shape[:2]
    start_x = IMAGE_AREA_X + PADDING
    start_y = IMAGE_AREA_Y + PADDING

    frame[start_y:start_y + h, start_x:start_x + w] = img

    # Нижняя полоса — прогресс
    cv2.rectangle(frame, (0, BOTTOM_BAR_Y), (WINDOW_WIDTH, WINDOW_HEIGHT), (240, 240, 240), -1)

    progress = (i + 1) / len(letters)
    bar_width = 600
    bar_x = (WINDOW_WIDTH - bar_width) // 2
    bar_y = BOTTOM_BAR_Y + 20

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (220, 220, 220), -1)
    filled = int(bar_width * progress)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 30), (70, 180, 70), -1)

    progress_text = f"{i + 1} / {len(letters)}"
    frame = draw_text_pil(frame, progress_text, bar_x + bar_width // 2 - 50, bar_y + 5,
                          font_progress, (0, 0, 0))

    # Подсказки
    hint_y = WINDOW_HEIGHT - 50
    frame = draw_text_pil(frame, "4 — назад    6 — вперёд    ESC / q — выйти", WINDOW_WIDTH // 2 - 300, hint_y,
                          font_hint, (100, 100, 100))

    cv2.imshow("Дактиль РЖЯ — Показ жестов", frame)

    key = cv2.waitKey(int(SHOW_TIME * 1000))

    if key == 27 or key == ord('q') or key == ord('Q'):
        print("Выход")
        break
    elif key == ord('4'):
        i = (i - 1) % len(letters)
    elif key == ord('6'):
        i = (i + 1) % len(letters)

print("\nПоказ завершён!")
cv2.destroyAllWindows()
"""
"""
#3

import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import ctypes  # Для надёжной проверки Ctrl
import pyperclip  # Для работы с буфером обмена (pip install pyperclip)

# === ШРИФТЫ ===
try:
    font_input = ImageFont.truetype("arial.ttf", 40)
    font_current = ImageFont.truetype("arial.ttf", 100)
    font_header = ImageFont.truetype("arial.ttf", 36)
    font_progress = ImageFont.truetype("arial.ttf", 30)
    font_hint = ImageFont.truetype("arial.ttf", 24)
    font_placeholder = ImageFont.truetype("arial.ttf", 36)
except IOError:
    font_input = ImageFont.load_default()
    font_current = ImageFont.load_default()
    font_header = ImageFont.load_default()
    font_progress = ImageFont.load_default()
    font_hint = ImageFont.load_default()
    font_placeholder = ImageFont.load_default()

def draw_text_pil(frame, text, x, y, font, color=(0,0,0), anchor=None, alpha=255):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    if alpha < 255:
        text_img = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((x, y), text, font=font, fill=color + (alpha,), anchor=anchor)
        img_pil = Image.alpha_composite(img_pil.convert("RGBA"), text_img).convert("RGB")
    else:
        draw.text((x, y), text, font=font, fill=color, anchor=anchor)
    return np.array(img_pil)

# === НАСТРОЙКИ ===
GESTURE_FOLDER = "alfavit_parts"

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

PANEL_WIDTH = 460
PANEL_HEIGHT = 540
PANEL_PADDING_H = 60
PANEL_PADDING_V = 20

top_y = PANEL_PADDING_V
bottom_y = top_y + PANEL_HEIGHT

left_x1 = PANEL_PADDING_H
left_x2 = left_x1 + PANEL_WIDTH

right_x1 = WINDOW_WIDTH - PANEL_PADDING_H - PANEL_WIDTH
right_x2 = right_x1 + PANEL_WIDTH

GESTURE_SIZE = (360, 480)

BOTTOM_Y = WINDOW_HEIGHT - 120

# === ЗАГРУЗКА ЖЕСТОВ ===
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, GESTURE_SIZE)

print(f"Загружено жестов: {len(gesture_images)} из {len(CLASSES)}")

# === СОСТОЯНИЕ ===
input_text = ""
current_index = 0

def get_valid_letters(text):
    return [c.upper() for c in text if c.upper() in gesture_images]

valid_letters = get_valid_letters(input_text)

# === ОСНОВНОЙ ЦИКЛ ===
while True:
    frame = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

    # ЛЕВАЯ ПАНЕЛЬ
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, bottom_y), (150, 200, 255), 2)
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, top_y + 60), (220, 240, 255), -1)
    frame = draw_text_pil(frame, "Ввод текста", left_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if not input_text:
        placeholder = "(печатайте на русском...)"
        center_x = left_x1 + PANEL_WIDTH // 2
        center_y = top_y + PANEL_HEIGHT // 2
        frame = draw_text_pil(frame, placeholder, center_x, center_y, font_placeholder,
                              (100, 100, 150), anchor="mm", alpha=150)

    # Введённый текст с переносом
    lines = []
    current_line = ""
    max_width = PANEL_WIDTH - 80
    for char in input_text:
        test_line = current_line + char
        text_width = font_input.getlength(test_line)
        if text_width > max_width and current_line:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    y_pos = top_y + 90
    for line in lines:
        if y_pos > bottom_y - 40:
            break
        frame = draw_text_pil(frame, line, left_x1 + 40, y_pos, font_input, (0, 0, 100))
        y_pos += 55

    # ПРАВАЯ ПАНЕЛЬ
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, bottom_y), (150, 200, 255), 2)
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, top_y + 60), (220, 240, 255), -1)
    frame = draw_text_pil(frame, "Жест РЖЯ", right_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if valid_letters and current_index < len(valid_letters):
        current_letter = valid_letters[current_index]
        center_x = right_x1 + PANEL_WIDTH // 2
        frame = draw_text_pil(frame, current_letter, center_x, bottom_y - 60,
                              font_current, (0, 0, 100), anchor="mm")

        img = gesture_images[current_letter]
        h, w = img.shape[:2]
        start_x = right_x1 + (PANEL_WIDTH - w) // 2
        start_y = top_y + 90
        frame[start_y:start_y + h, start_x:start_x + w] = img

    # НИЖНЯЯ ПОЛОСА
    cv2.rectangle(frame, (0, BOTTOM_Y), (WINDOW_WIDTH, WINDOW_HEIGHT), (220, 240, 255), -1)

    total = len(valid_letters)
    if total > 0:
        progress = (current_index + 1) / total
        bar_width = 760
        bar_x = (WINDOW_WIDTH - bar_width) // 2
        bar_y = BOTTOM_Y + 15

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 30), (150, 200, 255), -1)
        filled = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 30), (0, 0, 255), -1)

        progress_text = f"{current_index + 1} из {total}"
        prog_w = font_progress.getlength(progress_text)
        frame = draw_text_pil(frame, progress_text, bar_x + bar_width // 2 - prog_w // 2, bar_y + 4,
                              font_progress, (255, 255, 255))

    # Подсказка — две строки, по центру
    hint_line1 = "← → или 4/6 — переключить букву     |     Backspace — стереть"
    hint_line2 = "Ctrl+C — копировать     |     Ctrl+V — вставить     |     ESC — выйти"

    w1 = font_hint.getlength(hint_line1)
    w2 = font_hint.getlength(hint_line2)

    frame = draw_text_pil(frame, hint_line1, (WINDOW_WIDTH - w1) // 2, BOTTOM_Y + 55, font_hint, (0, 0, 100))
    frame = draw_text_pil(frame, hint_line2, (WINDOW_WIDTH - w2) // 2, BOTTOM_Y + 85, font_hint, (0, 0, 100))

    cv2.imshow("Дактильная азбука РЖЯ", frame)

    # === ОБРАБОТКА КЛАВИШ ===
    key = cv2.waitKey(10) & 0xFF  # Небольшая задержка для стабильности

    if key == 27:  # ESC
        break
    elif key == 8 and input_text:  # Backspace
        input_text = input_text[:-1]
    elif key == 13:  # Enter
        input_text += '\n'

    # === НАДЁЖНАЯ ПРОВЕРКА CTRL ===
    ctrl_down = ctypes.windll.user32.GetKeyState(0x11) & 0x8000  # Бит 15 = зажата

    if ctrl_down:
        if key == ord('c') or key == ord('C'):
            pyperclip.copy(input_text)
            continue
        elif key == ord('v') or key == ord('V'):
            try:
                clipboard_text = pyperclip.paste()
                input_text += clipboard_text
            except:
                pass
            continue

    # Навигация (приоритет)
    if key in [81, 83, ord('4'), ord('6')]:
        if key in [81, ord('4')]:
            if valid_letters:
                current_index = (current_index - 1) % len(valid_letters)
        else:
            if valid_letters:
                current_index = (current_index + 1) % len(valid_letters)
        continue

    # Обычный ввод символов
    if key != 255 and key != 0:
        try:
            char = bytes([key]).decode('windows-1251')
            if char.isprintable() or char == ' ':
                input_text += char
        except:
            pass

    # Обновление списка букв
    valid_letters = get_valid_letters(input_text)
    current_index = min(current_index, len(valid_letters) - 1) if valid_letters else 0

cv2.destroyAllWindows()
print("Программа завершена.")
"""


"""
4
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
from tkinter import filedialog

# ================== ШРИФТЫ ==================
try:
    font_input = ImageFont.truetype("arial.ttf", 34)
    font_current = ImageFont.truetype("arial.ttf", 96)
    font_header = ImageFont.truetype("arial.ttf", 32)
    font_hint = ImageFont.truetype("arial.ttf", 20)
except:
    font_input = font_current = font_header = font_hint = ImageFont.load_default()

# ================== НАСТРОЙКИ ==================
GESTURE_FOLDER = "alfavit_parts"
CLASSES = [
    'Ё','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л',
    'М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш',
    'Щ','Ъ','Ы','Ь','Э','Ю','Я'
]
ALLOWED_CHARS = set(
    "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ "
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя\n"
)
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
PANEL_WIDTH, PANEL_HEIGHT = 460, 600
PANEL_PADDING_H, PANEL_PADDING_V = 60, 20
TEXT_PADDING_X = 50
TEXT_PADDING_Y = 100
LINE_HEIGHT = 50
BOTTOM_Y = WINDOW_HEIGHT - 100
GESTURE_SIZE = (360, 480)

top_y = PANEL_PADDING_V
bottom_y = top_y + PANEL_HEIGHT
TEXT_MAX_Y = bottom_y - 20
left_x1 = PANEL_PADDING_H
left_x2 = left_x1 + PANEL_WIDTH
right_x1 = WINDOW_WIDTH - PANEL_PADDING_H - PANEL_WIDTH
right_x2 = right_x1 + PANEL_WIDTH

# ================== ЖЕСТЫ ==================
gesture_images = {}
for l in CLASSES:
    p = os.path.join(GESTURE_FOLDER, f"{l}.png")
    if os.path.exists(p):
        img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[l] = cv2.resize(img, GESTURE_SIZE)

# ================== СОСТОЯНИЕ ==================
input_text = ""
current_index = 0
history = []
cursor_visible = True
cursor_timer = 0

# ================== ФУНКЦИИ ==================
def draw_text(frame, text, x, y, font, color, anchor=None, alpha=255):
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    draw.text((x, y), text, font=font, fill=color + (alpha,), anchor=anchor)
    return np.array(Image.alpha_composite(img, overlay).convert("RGB"))

def get_valid_letters(text):
    return [c.upper() for c in text if c.upper() in gesture_images]

def save_history():
    global history
    history.append(input_text)
    if len(history) > 50:
        history.pop(0)

def load_text():
    global input_text, current_index
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not path:
        return
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    input_text = "".join(c for c in content if c in ALLOWED_CHARS)
    current_index = len(input_text)

def draw_text_with_cursor(frame, x, y, text, font, max_width, cursor_pos, cursor_visible=True):
    line_x, line_y = x, y
    letter_counter = 0
    for ch in text:
        if line_y > TEXT_MAX_Y:
            break
        if ch == "\n":
            line_x = x
            line_y += LINE_HEIGHT
            continue
        w = font.getlength(ch)
        if line_x + w > x + max_width:
            line_x = x
            line_y += LINE_HEIGHT
        frame = draw_text(frame, ch, line_x, line_y, font, (0,0,0))
        if letter_counter == cursor_pos and cursor_visible:
            cv2.line(frame, (int(line_x + w/2), int(line_y)), (int(line_x + w/2), int(line_y + LINE_HEIGHT - 10)), (0,0,0), 2)
        line_x += w
        letter_counter += 1
    # Курсор в конце текста
    if cursor_pos == letter_counter and cursor_visible:
        cv2.line(frame, (int(line_x), int(line_y)), (int(line_x), int(line_y + LINE_HEIGHT - 10)), (0,0,0), 2)
    return frame

# ================== ОСНОВНОЙ ЦИКЛ ==================
while True:
    frame = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255

    # === ЗАЩИТА КУРСОРА И ОБНОВЛЕНИЕ ВАЛИДНЫХ БУКВ ===
    current_index = max(0, min(current_index, len(input_text)))
    valid_letters = get_valid_letters(input_text)

    # ==== ЛЕВАЯ ПАНЕЛЬ ====
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, bottom_y), (150,200,255), 2)
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, top_y+60), (220,240,255), -1)
    frame = draw_text(frame, "Ввод текста", left_x1+30, top_y+15, font_header, (0,0,100))

    # Мигание курсора
    cursor_timer += 1
    if cursor_timer >= 25:
        cursor_timer = 0
        cursor_visible = not cursor_visible

    if not input_text.strip():
        yy = top_y + TEXT_PADDING_Y
        for line in [
            "Введите текст",
            "Enter — новая строка",
            "Backspace — удалить символ",
            "Delete — очистить весь текст",
            "Ctrl+Z — отмена",
            "L — загрузить текст"
        ]:
            frame = draw_text(frame, line, left_x1 + PANEL_WIDTH//2, yy, font_hint, (80,80,120), anchor="mm", alpha=140)
            yy += 28
    else:
        frame = draw_text_with_cursor(frame, left_x1 + TEXT_PADDING_X, top_y + TEXT_PADDING_Y,
                                      input_text, font_input,
                                      PANEL_WIDTH - 2*TEXT_PADDING_X, current_index, cursor_visible)

    # ==== ПРАВАЯ ПАНЕЛЬ ====
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, bottom_y), (150,200,255), 2)
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, top_y+60), (220,240,255), -1)
    frame = draw_text(frame, "Жест РЖЯ", right_x1+30, top_y+15, font_header, (0,0,100))

    if valid_letters:
        # Текущая буква для показа жеста (циклически)
        letter_index = current_index if current_index < len(valid_letters) else 0
        letter = valid_letters[letter_index]
        frame = draw_text(frame, letter, right_x1 + PANEL_WIDTH//2, bottom_y-60, font_current, (0,0,120), anchor="mm")
        img = gesture_images[letter]
        h, w = img.shape[:2]
        frame[top_y+90:top_y+90+h, right_x1+(PANEL_WIDTH-w)//2:right_x1+(PANEL_WIDTH-w)//2+w] = img

    # ==== ПОДСКАЗКА СНИЗУ ====
    frame = draw_text(frame, "← → или 4 / 6 — переключить букву | ESC — выход",
                      WINDOW_WIDTH//2, BOTTOM_Y+30, font_hint, (0,0,90), anchor="mm")

    cv2.imshow("Дактильная азбука РЖЯ", frame)
    key = cv2.waitKey(15) & 0xFF

    # ==== УПРАВЛЕНИЕ ====
    if key == 27:  # ESC
        break
    elif key == 8:  # Backspace
        if input_text and current_index > 0:
            save_history()
            input_text = input_text[:current_index-1] + input_text[current_index:]
            current_index -= 1
    elif key == 127:  # Delete — полная очистка
        if input_text:  # только если есть текст
            save_history()
            input_text = ""
            current_index = 0
            # valid_letters обновится автоматически в начале следующей итерации
    elif key == 26:  # Ctrl+Z — Undo
        if history:
            input_text = history.pop()
            current_index = len(input_text)
    elif key == 13:  # Enter
        save_history()
        input_text = input_text[:current_index] + "\n" + input_text[current_index:]
        current_index += 1
    elif key in (ord('l'), ord('L')):  # загрузка файла
        save_history()
        load_text()
    elif key in (81, 83, ord('4'), ord('6')) and valid_letters:  # ← → или 4/6
        if key in (81, ord('4')):  # влево
            current_index = (current_index - 1) % len(valid_letters)
        else:  # вправо
            current_index = (current_index + 1) % len(valid_letters)
    elif key != 255:
        try:
            ch = bytes([key]).decode("windows-1251")
            if ch in ALLOWED_CHARS:
                save_history()
                input_text = input_text[:current_index] + ch + input_text[current_index:]
                current_index += 1
        except:
            pass

cv2.destroyAllWindows()
"""
"""
#6
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
from tkinter import scrolledtext

# ===================== ШРИФТЫ =====================
try:
    font_text     = ImageFont.truetype("arial.ttf", 48)   # Крупный текст слева
    font_header   = ImageFont.truetype("arial.ttf", 38)
    font_letter   = ImageFont.truetype("arial.ttf", 100)
    font_progress = ImageFont.truetype("arial.ttf", 34)
    font_button   = ImageFont.truetype("arial.ttf", 24)
    font_hint     = ImageFont.truetype("arial.ttf", 24)
except:
    font_text = font_header = font_letter = font_progress = font_button = font_hint = ImageFont.load_default()

def draw_text(frame, text, x, y, font, color=(0,0,0), anchor=None):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color, anchor=anchor)
    return np.array(pil_img)

# ===================== НАСТРОЙКИ =====================
GESTURE_FOLDER = "alfavit_parts"
CLASSES = ['Ё','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']

W, H = 1280, 720
PW, PH = 460, 540
PAD_H, PAD_V = 60, 20

top_y = PAD_V
bottom_y = top_y + PH
left_x1 = PAD_H
left_x2 = left_x1 + PW
right_x1 = W - PAD_H - PW
right_x2 = right_x1 + PW
BOTTOM_BAR_Y = H - 140

LINE_SPACING = 62

# ===================== ЗАГРУЗКА ЖЕСТОВ =====================
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, (380, 500))

print(f"Загружено жестов: {len(gesture_images)}")

# ===================== СОСТОЯНИЕ =====================
input_text = ""
current_index = 0
valid_letters = []

# ===================== ОКНО ВВОДА — ИСПРАВЛЕНО =====================
def open_text_editor():
    global input_text, valid_letters, current_index
    editor = tk.Tk()
    editor.title("Ввод / Редактирование текста")
    editor.geometry("900x600")
    editor.configure(bg="#f0f0f0")

    tk.Label(editor, text="Введите или отредактируйте текст на русском языке", font=("Arial", 16), bg="#f0f0f0").pack(pady=15)

    text_widget = scrolledtext.ScrolledText(editor, wrap=tk.WORD, font=("Arial", 18), padx=20, pady=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
    text_widget.insert("1.0", input_text)
    text_widget.focus_set()

    def save():
        global input_text, valid_letters, current_index
        input_text = text_widget.get("1.0", tk.END).rstrip()
        valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
        current_index = 0
        editor.destroy()

    btn_frame = tk.Frame(editor, bg="#f0f0f0")
    btn_frame.pack(pady=15)
    tk.Button(btn_frame, text="Сохранить и применить", font=("Arial", 14), width=30, bg="#4CAF50", fg="white", command=save).pack()

    editor.bind("<Control-s>", lambda e: save())
    editor.bind("<Return>", lambda e: save())
    editor.protocol("WM_DELETE_WINDOW", save)  # Крестик тоже сохраняет

    editor.mainloop()
    # После закрытия окна — текст сразу обновится в следующем кадре OpenCV

# Очистка текста
def clear_text():
    global input_text, current_index
    input_text = ""
    current_index = 0

# ===================== ОСНОВНОЙ ЦИКЛ =====================
cv2.namedWindow("Дактильная азбука РЖЯ")
cv2.resizeWindow("Дактильная азбука РЖЯ", W, H)

while True:
    frame = np.ones((H, W, 3), np.uint8) * 255

    valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
    total = len(valid_letters)

    # === ЛЕВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Введённый текст", left_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if input_text:
        words = input_text.split()
        lines = []
        line = ""
        max_w = PW - 80
        for word in words:
            test = line + (" " + word if line else word)
            if font_text.getlength(test) > max_w and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        y = top_y + 90
        for line in lines:
            if y > bottom_y - 60:
                break
            frame = draw_text(frame, line, left_x1 + 40, y, font_text, (0, 0, 100))
            y += LINE_SPACING
    else:
        frame = draw_text(frame, "Нажмите 1 или 2 для ввода", left_x1 + 40, top_y + 220, font_hint, (120, 120, 160))

    # === ПРАВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Жест РЖЯ", right_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if total > 0:
        letter = valid_letters[current_index % total]
        if letter in gesture_images:
            img = gesture_images[letter]
            h, w = img.shape[:2]
            sx = right_x1 + (PW - w) // 2
            sy = top_y + 90
            frame[sy:sy + h, sx:sx + w] = img

        frame = draw_text(frame, letter, right_x1 + PW // 2, bottom_y - 80, font_letter, (0, 0, 160), anchor="mm")
        progress_text = f"{(current_index % total) + 1} / {total}"
        frame = draw_text(frame, progress_text, right_x1 + PW // 2, bottom_y - 160, font_progress, (0, 0, 120), anchor="mm")
    else:
        frame = draw_text(frame, "Нет букв РЖЯ", right_x1 + PW // 2, top_y + 250, font_hint, (100, 100, 100), anchor="mm")

    # === НИЖНЯЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (0, BOTTOM_BAR_Y), (W, H), (210, 235, 255), -1)

    # Подсказка
    hint = "1 — ввести | 2 — редактировать | 0 — очистить | 4/6 или ←→ — жест | ESC — выход"
    frame = draw_text(frame, hint, W // 2, BOTTOM_BAR_Y + 15, font_hint, (0, 0, 120), anchor="mm")

    # Кнопки — компактные
    btn_w, btn_h = 260, 55
    btn_y = BOTTOM_BAR_Y + 55
    spacing = 40

    # Кнопка 1
    btn1_x = 80
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (70, 130, 255), -1)
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (100, 160, 255), 5)
    frame = draw_text(frame, "1. Ввести текст", btn1_x + (btn_w - font_button.getlength("1. Ввести текст")) // 2, btn_y + 16, font_button, (255, 255, 255))

    # Кнопка 2
    btn2_x = btn1_x + btn_w + spacing
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (70, 200, 100), -1)
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (100, 240, 130), 5)
    frame = draw_text(frame, "2. Редактировать", btn2_x + (btn_w - font_button.getlength("2. Редактировать")) // 2, btn_y + 16, font_button, (255, 255, 255))

    # Кнопка 0 — Очистить
    btn0_x = btn2_x + btn_w + spacing
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (200, 100, 100), -1)
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (240, 140, 140), 5)
    frame = draw_text(frame, "0. Очистить", btn0_x + (btn_w - font_button.getlength("0. Очистить")) // 2, btn_y + 16, font_button, (255, 255, 255))

    # Кнопка ESC — Выход
    btn3_x = btn0_x + btn_w + spacing
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 120, 80), -1)
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 160, 120), 5)
    frame = draw_text(frame, "ESC. Выход", btn3_x + (btn_w - font_button.getlength("ESC. Выход")) // 2, btn_y + 16, font_button, (255, 255, 255))

    cv2.imshow("Дактильная азбука РЖЯ", frame)

    key = cv2.waitKey(10) & 0xFF

    if key in (27, ord('3')):
        break
    elif key == ord('1') or key == ord('2'):
        open_text_editor()
    elif key == ord('0'):
        clear_text()
    elif key in (ord('4'), 81):
        if total > 0:
            current_index -= 1
    elif key in (ord('6'), 83):
        if total > 0:
            current_index += 1

cv2.destroyAllWindows()
print("Программа завершена.")
"""
"""
import cv2
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
from tkinter import scrolledtext
import pyperclip  # pip install pyperclip

# ===================== ШРИФТЫ =====================
try:
    font_text     = ImageFont.truetype("arial.ttf", 38)
    font_header   = ImageFont.truetype("arial.ttf", 38)
    font_progress = ImageFont.truetype("arial.ttf", 36)
    font_button   = ImageFont.truetype("arial.ttf", 24)
    font_hint     = ImageFont.truetype("arial.ttf", 22)
except:
    font_text = font_header = font_progress = font_button = font_hint = ImageFont.load_default()

def draw_text(frame, text, x, y, font, color=(0,0,0), anchor=None):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color, anchor=anchor)
    return np.array(pil_img)

# ===================== НАСТРОЙКИ =====================
GESTURE_FOLDER = "alfavit_parts"
CLASSES = ['Ё','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']

W, H = 1280, 720
PW, PH = 460, 540
PAD_H, PAD_V = 60, 20

top_y = PAD_V
bottom_y = top_y + PH
left_x1 = PAD_H
left_x2 = left_x1 + PW
right_x1 = W - PAD_H - PW
right_x2 = right_x1 + PW
BOTTOM_BAR_Y = H - 140

LINE_SPACING = 48

# УМЕНЬШЕННЫЙ РАЗМЕР ЖЕСТА — чтобы не выходил за границы панели
GESTURE_SIZE = (320, 420)

# ===================== ЗАГРУЗКА ЖЕСТОВ =====================
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, GESTURE_SIZE)

print(f"Загружено жестов: {len(gesture_images)}")

# ===================== СОСТОЯНИЕ =====================
input_text = ""
current_index = 0
valid_letters = []
text_scroll_offset = 0

# ===================== ОКНО ВВОДА =====================
def open_text_editor():
    global input_text, valid_letters, current_index, text_scroll_offset
    editor = tk.Tk()
    editor.title("Ввод / Редактирование текста")
    editor.geometry("900x600")
    editor.configure(bg="#f0f0f0")

    tk.Label(editor, text="Введите или отредактируйте текст на русском языке", font=("Arial", 16), bg="#f0f0f0").pack(pady=15)

    text_widget = scrolledtext.ScrolledText(editor, wrap=tk.WORD, font=("Arial", 18), padx=20, pady=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
    text_widget.insert("1.0", input_text)
    text_widget.focus_set()

    def save():
        global input_text, valid_letters, current_index, text_scroll_offset
        input_text = text_widget.get("1.0", tk.END).rstrip()
        valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
        current_index = 0
        text_scroll_offset = 0
        editor.destroy()

    tk.Button(editor, text="Сохранить и применить", font=("Arial", 14), width=30, bg="#4CAF50", fg="white", command=save).pack(pady=15)
    editor.bind("<Control-s>", lambda e: save())
    editor.bind("<Return>", lambda e: save())
    editor.protocol("WM_DELETE_WINDOW", save)

    editor.mainloop()

# Очистка текста
def clear_text():
    global input_text, current_index, text_scroll_offset
    input_text = ""
    current_index = 0
    text_scroll_offset = 0

# ===================== ОСНОВНОЙ ЦИКЛ =====================
cv2.namedWindow("Дактильная азбука РЖЯ")
cv2.resizeWindow("Дактильная азбука РЖЯ", W, H)

while True:
    frame = np.ones((H, W, 3), np.uint8) * 255

    valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
    total = len(valid_letters)

    # === ЛЕВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Введённый текст", left_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if input_text:
        words = input_text.split()
        lines = []
        line = ""
        max_w = PW - 80
        for word in words:
            test = line + (" " + word if line else word)
            if font_text.getlength(test) > max_w and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        max_visible_lines = (PH - 140) // LINE_SPACING
        text_scroll_offset = max(0, min(text_scroll_offset, max(0, len(lines) - max_visible_lines)))

        y = top_y + 90
        for line in lines[text_scroll_offset:text_scroll_offset + max_visible_lines]:
            frame = draw_text(frame, line, left_x1 + 40, y, font_text, (0, 0, 100))
            y += LINE_SPACING
    else:
        frame = draw_text(frame, "Нажмите 1 или 2 для ввода", left_x1 + 40, top_y + 220, font_hint, (120, 120, 160))

    # === ПРАВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Жест РЖЯ", right_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if total > 0:
        letter = valid_letters[current_index % total]
        if letter in gesture_images:
            img = gesture_images[letter]
            h, w = img.shape[:2]
            sx = right_x1 + (PW - w) // 2
            sy = top_y + 120
            frame[sy:sy + h, sx:sx + w] = img

        progress_text = f"{(current_index % total) + 1} / {total}"
        frame = draw_text(frame, progress_text, right_x1 + PW // 2, top_y + 80, font_progress, (0, 0, 150), anchor="mm")
    else:
        frame = draw_text(frame, "Нет букв РЖЯ", right_x1 + PW // 2, top_y + 250, font_hint, (100, 100, 100), anchor="mm")

    # === НИЖНЯЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (0, BOTTOM_BAR_Y), (W, H), (210, 235, 255), -1)

    hint = "1-ввод | 2-редакт | 0-очистить | 4/6-жест | 8/5-прокрутка | V-вставить | ESC-выход"
    frame = draw_text(frame, hint, W // 2, BOTTOM_BAR_Y + 18, font_hint, (0, 0, 120), anchor="mm")

    # Кнопки
    btn_w, btn_h = 260, 55
    btn_y = BOTTOM_BAR_Y + 55
    spacing = 40

    btn1_x = 80
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (70, 130, 255), -1)
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (100, 160, 255), 5)
    frame = draw_text(frame, "1. Ввести текст", btn1_x + (btn_w - font_button.getlength("1. Ввести текст")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn2_x = btn1_x + btn_w + spacing
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (70, 200, 100), -1)
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (100, 240, 130), 5)
    frame = draw_text(frame, "2. Редактировать", btn2_x + (btn_w - font_button.getlength("2. Редактировать")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn0_x = btn2_x + btn_w + spacing
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (200, 100, 100), -1)
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (240, 140, 140), 5)
    frame = draw_text(frame, "0. Очистить", btn0_x + (btn_w - font_button.getlength("0. Очистить")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn3_x = btn0_x + btn_w + spacing
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 120, 80), -1)
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 160, 120), 5)
    frame = draw_text(frame, "ESC. Выход", btn3_x + (btn_w - font_button.getlength("ESC. Выход")) // 2, btn_y + 16, font_button, (255, 255, 255))

    cv2.imshow("Дактильная азбука РЖЯ", frame)

    key = cv2.waitKey(10) & 0xFF

    if key in (27, ord('3')):
        break
    elif key == ord('1') or key == ord('2'):
        open_text_editor()
    elif key == ord('0'):
        clear_text()
    elif key == ord('8'):
        text_scroll_offset = max(0, text_scroll_offset - 1)
    elif key == ord('5'):
        if 'lines' in locals():
            max_visible = (PH - 140) // LINE_SPACING
            text_scroll_offset = min(text_scroll_offset + 1, max(0, len(lines) - max_visible))
    elif key in (ord('v'), ord('V'), ord('м'), ord('М')):
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                input_text += clipboard_text
                valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
                current_index = 0
                text_scroll_offset = 0
        except:
            pass
    elif key in (ord('4'), 81):
        if total > 0:
            current_index -= 1
    elif key in (ord('6'), 83):
        if total > 0:
            current_index += 1

cv2.destroyAllWindows()
print("Программа завершена.")
"""
import cv2
import os
import sys
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import tkinter as tk
from tkinter import scrolledtext
import pyperclip


def resource_path(relative_path):
    """Корректный путь для .py и .exe (PyInstaller)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ===================== ШРИФТЫ =====================
try:
    font_text     = ImageFont.truetype("arial.ttf", 38)
    font_header   = ImageFont.truetype("arial.ttf", 38)
    font_progress = ImageFont.truetype("arial.ttf", 36)
    font_button   = ImageFont.truetype("arial.ttf", 24)
    font_hint     = ImageFont.truetype("arial.ttf", 22)
except:
    font_text = font_header = font_progress = font_button = font_hint = ImageFont.load_default()

def draw_text(frame, text, x, y, font, color=(0,0,0), anchor=None):
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color, anchor=anchor)
    return np.array(pil_img)

# ===================== НАСТРОЙКИ =====================
GESTURE_FOLDER = resource_path("alfavit_parts")
CLASSES = ['Ё','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','Н','О','П','Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я']

W, H = 1280, 720
PW, PH = 460, 540
PAD_H, PAD_V = 60, 20

top_y = PAD_V
bottom_y = top_y + PH
left_x1 = PAD_H
left_x2 = left_x1 + PW
right_x1 = W - PAD_H - PW
right_x2 = right_x1 + PW
BOTTOM_BAR_Y = H - 140

LINE_SPACING = 48

# УМЕНЬШЕННЫЙ РАЗМЕР ЖЕСТА — чтобы не выходил за границы панели
GESTURE_SIZE = (320, 420)

# ===================== ЗАГРУЗКА ЖЕСТОВ =====================
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gesture_images[letter] = cv2.resize(img, GESTURE_SIZE)
        else:
            print("❌ Не удалось загрузить:", path)
    else:
        print("❌ Файл не найден:", path)

print(f"Загружено жестов: {len(gesture_images)}")

# ===================== СОСТОЯНИЕ =====================
input_text = ""
current_index = 0
valid_letters = []
text_scroll_offset = 0

# ===================== ОКНО ВВОДА =====================
def open_text_editor():
    global input_text, valid_letters, current_index, text_scroll_offset
    editor = tk.Tk()
    editor.title("Ввод / Редактирование текста")
    editor.geometry("900x600")
    editor.configure(bg="#f0f0f0")

    tk.Label(editor, text="Введите или отредактируйте текст на русском языке", font=("Arial", 16), bg="#f0f0f0").pack(pady=15)

    text_widget = scrolledtext.ScrolledText(editor, wrap=tk.WORD, font=("Arial", 18), padx=20, pady=20)
    text_widget.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
    text_widget.insert("1.0", input_text)
    text_widget.focus_set()

    def save():
        global input_text, valid_letters, current_index, text_scroll_offset
        input_text = text_widget.get("1.0", tk.END).rstrip()
        valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
        current_index = 0
        text_scroll_offset = 0
        editor.destroy()

    tk.Button(editor, text="Сохранить и применить", font=("Arial", 14), width=30, bg="#4CAF50", fg="white", command=save).pack(pady=15)
    editor.bind("<Control-s>", lambda e: save())
    editor.bind("<Return>", lambda e: save())
    editor.protocol("WM_DELETE_WINDOW", save)

    editor.mainloop()

# Очистка текста
def clear_text():
    global input_text, current_index, text_scroll_offset
    input_text = ""
    current_index = 0
    text_scroll_offset = 0

# ===================== ОСНОВНОЙ ЦИКЛ =====================
cv2.namedWindow("Дактильная азбука РЖЯ")
cv2.resizeWindow("Дактильная азбука РЖЯ", W, H)

while True:
    frame = np.ones((H, W, 3), np.uint8) * 255

    valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
    total = len(valid_letters)

    # === ЛЕВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (left_x1, top_y), (left_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Введённый текст", left_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if input_text:
        words = input_text.split()
        lines = []
        line = ""
        max_w = PW - 80
        for word in words:
            test = line + (" " + word if line else word)
            if font_text.getlength(test) > max_w and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)

        max_visible_lines = (PH - 140) // LINE_SPACING
        text_scroll_offset = max(0, min(text_scroll_offset, max(0, len(lines) - max_visible_lines)))

        y = top_y + 90
        for line in lines[text_scroll_offset:text_scroll_offset + max_visible_lines]:
            frame = draw_text(frame, line, left_x1 + 40, y, font_text, (0, 0, 100))
            y += LINE_SPACING
    else:
        frame = draw_text(frame, "Нажмите 1 или 2 для ввода", left_x1 + 40, top_y + 220, font_hint, (120, 120, 160))

    # === ПРАВАЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, bottom_y), (140, 190, 255), 4)
    cv2.rectangle(frame, (right_x1, top_y), (right_x2, top_y + 60), (210, 235, 255), -1)
    frame = draw_text(frame, "Жест РЖЯ", right_x1 + 30, top_y + 15, font_header, (0, 0, 100))

    if total > 0:
        letter = valid_letters[current_index % total]
        if letter in gesture_images:
            img = gesture_images[letter]
            h, w = img.shape[:2]
            sx = right_x1 + (PW - w) // 2
            sy = top_y + 120
            frame[sy:sy + h, sx:sx + w] = img

        progress_text = f"{(current_index % total) + 1} / {total}"
        frame = draw_text(frame, progress_text, right_x1 + PW // 2, top_y + 80, font_progress, (0, 0, 150), anchor="mm")
    else:
        frame = draw_text(frame, "Нет букв РЖЯ", right_x1 + PW // 2, top_y + 250, font_hint, (100, 100, 100), anchor="mm")

    # === НИЖНЯЯ ПАНЕЛЬ ===
    cv2.rectangle(frame, (0, BOTTOM_BAR_Y), (W, H), (210, 235, 255), -1)

    hint = "1-ввод | 2-редакт | 0-очистить | 4/6-жест | 8/5-прокрутка | V-вставить | ESC-выход"
    frame = draw_text(frame, hint, W // 2, BOTTOM_BAR_Y + 18, font_hint, (0, 0, 120), anchor="mm")

    # Кнопки
    btn_w, btn_h = 260, 55
    btn_y = BOTTOM_BAR_Y + 55
    spacing = 40

    btn1_x = 80
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (70, 130, 255), -1)
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + btn_w, btn_y + btn_h), (100, 160, 255), 5)
    frame = draw_text(frame, "1. Ввести текст", btn1_x + (btn_w - font_button.getlength("1. Ввести текст")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn2_x = btn1_x + btn_w + spacing
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (70, 200, 100), -1)
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + btn_w, btn_y + btn_h), (100, 240, 130), 5)
    frame = draw_text(frame, "2. Редактировать", btn2_x + (btn_w - font_button.getlength("2. Редактировать")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn0_x = btn2_x + btn_w + spacing
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (200, 100, 100), -1)
    cv2.rectangle(frame, (btn0_x, btn_y), (btn0_x + btn_w, btn_y + btn_h), (240, 140, 140), 5)
    frame = draw_text(frame, "0. Очистить", btn0_x + (btn_w - font_button.getlength("0. Очистить")) // 2, btn_y + 16, font_button, (255, 255, 255))

    btn3_x = btn0_x + btn_w + spacing
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 120, 80), -1)
    cv2.rectangle(frame, (btn3_x, btn_y), (btn3_x + btn_w, btn_y + btn_h), (255, 160, 120), 5)
    frame = draw_text(frame, "ESC. Выход", btn3_x + (btn_w - font_button.getlength("ESC. Выход")) // 2, btn_y + 16, font_button, (255, 255, 255))

    cv2.imshow("Дактильная азбука РЖЯ", frame)

    key = cv2.waitKey(10) & 0xFF

    if key in (27, ord('3')):
        break
    elif key == ord('1') or key == ord('2'):
        open_text_editor()
    elif key == ord('0'):
        clear_text()
    elif key == ord('8'):
        text_scroll_offset = max(0, text_scroll_offset - 1)
    elif key == ord('5'):
        if 'lines' in locals():
            max_visible = (PH - 140) // LINE_SPACING
            text_scroll_offset = min(text_scroll_offset + 1, max(0, len(lines) - max_visible))
    elif key in (ord('v'), ord('V'), ord('м'), ord('М')):
        try:
            clipboard_text = pyperclip.paste()
            if clipboard_text:
                input_text += clipboard_text
                valid_letters = [c.upper() for c in input_text if c.upper() in gesture_images]
                current_index = 0
                text_scroll_offset = 0
        except:
            pass
    elif key in (ord('4'), 81):
        if total > 0:
            current_index -= 1
    elif key in (ord('6'), 83):
        if total > 0:
            current_index += 1

cv2.destroyAllWindows()
print("Программа завершена.")
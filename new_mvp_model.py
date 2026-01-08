""""
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import SignTransformer
import os

# === КИРИЛЛИЦА ДЛЯ OpenCV ===
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype("arial.ttf", 60)
small_font = ImageFont.truetype("arial.ttf", 40)


def draw_text_cyr(frame, text, x, y, font_obj, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font_obj, fill=color)
    return np.array(img_pil)


def draw_wrapped_text(frame, text, start_x, start_y, font_obj, color, max_width):
    if not text:
        return frame

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    lines = text.split('\n')
    y = start_y
    line_height = font_obj.size + 15

    for i, line in enumerate(lines):
        words = line.split()
        current_words = []

        for word in words:
            test_line = " ".join(current_words + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font_obj)
            w = bbox[2] - bbox[0]

            if w <= max_width:
                current_words.append(word)
            else:
                if current_words:
                    draw.text((start_x, y), " ".join(current_words), font=font_obj, fill=color)
                    y += line_height
                    current_words = [word]
                else:
                    draw.text((start_x, y), word, font=font_obj, fill=color)
                    y += line_height
                    current_words = []

        if current_words:
            draw.text((start_x, y), " ".join(current_words), font=font_obj, fill=color)
            y += line_height

        if i < len(lines) - 1:
            y += 20

    return np.array(img_pil)


# ======== CONFIG ========
MODEL_PATH = "bukva/best_model_fixed.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
USE_MOTION = True

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

model = SignTransformer(input_dim=189).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)  # Исправлено предупреждение
model.load_state_dict(state)
model.eval()

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_landmarks(landmarks):
    if not landmarks:
        return np.zeros(63, dtype=np.float32)
    lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = lm_array[0]
    normalized = lm_array - wrist
    max_dist = np.linalg.norm(normalized, axis=1).max()
    if max_dist > 0:
        normalized /= max_dist
    return normalized.flatten()


def add_motion_features(seq):
    seq = seq.astype(np.float32)
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    return np.concatenate([seq, vel, acc], axis=1)


def per_sample_normalize(x):
    x = x.astype(np.float32)
    mx = np.max(np.abs(x))
    return x / mx if mx > 0 else x


# === ЗАГРУЗКА КАРТИНОК ЖЕСТОВ ===
GESTURE_FOLDER = "alfavit_parts"  # Твоя папка с рисунками
gesture_images = {}
for letter in CLASSES:
    path = os.path.join(GESTURE_FOLDER, f"{letter}.png")
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            # Автоматически подгоняем размер (чтобы все были одинаковые)
            gesture_images[letter] = cv2.resize(img, (450, 650))

# === ПЕРЕМЕННЫЕ ДЛЯ РЕЖИМА ЧТЕНИЯ ===
read_letters = []
read_index = 0
read_timer = 0
READ_DELAY = 45  # ~1.5 сек на букву

fixed_background = None  # Фиксированный фон для показа

# === РЕЖИМЫ ===
MODE_INPUT = 1
MODE_READ = 2
MODE_PAUSE = 3

current_mode = MODE_INPUT

# === Основные переменные ===
buffer = []
text = ""
current_pred = None
confirmed_letter = None
stable_frames = 0
CONFIRM_FRAMES = 15

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

LEFT_MARGIN = 50
RIGHT_MARGIN = 50
TOP_MARGIN = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    max_text_width = frame_width - LEFT_MARGIN - RIGHT_MARGIN

    pred_text = ""

    # === РЕЖИМ ВВОДА ===
    if current_mode == MODE_INPUT:
        fixed_background = frame.copy()  # Сохраняем фон на случай перехода

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            features = normalize_landmarks(hand.landmark)
            buffer.append(features)
            if len(buffer) > SEQ_LEN:
                buffer.pop(0)

            if len(buffer) == SEQ_LEN:
                seq = np.array(buffer, dtype=np.float32)
                if USE_MOTION:
                    seq = add_motion_features(seq)
                seq = per_sample_normalize(seq)
                seq = seq[np.newaxis, ...]

                with torch.no_grad():
                    input_tensor = torch.from_numpy(seq).to(DEVICE)
                    logits = model(input_tensor)
                    pred_id = logits.argmax(1).item()
                    pred_text = CLASSES[pred_id]

                    if pred_text == current_pred:
                        stable_frames += 1
                    else:
                        current_pred = pred_text
                        stable_frames = 1

                    if stable_frames >= CONFIRM_FRAMES and pred_text != confirmed_letter:
                        text += pred_text
                        confirmed_letter = pred_text
                        stable_frames = 0
        else:
            buffer.clear()
            current_pred = None
            confirmed_letter = None
            stable_frames = 0
            pred_text = ""

    # === РЕЖИМ ЧТЕНИЯ ===
    elif current_mode == MODE_READ:
        display_frame = fixed_background.copy()

        if read_letters and read_index < len(read_letters):
            read_timer += 1
            if read_timer >= READ_DELAY:
                read_timer = 0
                read_index += 1

            if read_index < len(read_letters):
                current_letter = read_letters[read_index]

                if current_letter in gesture_images:
                    img = gesture_images[current_letter]
                    h, w = img.shape[:2]
                    start_x = frame_width - w - 50
                    start_y = 100

                    cv2.rectangle(display_frame, (start_x - 20, start_y - 20),
                                  (start_x + w + 20, start_y + h + 100), (255, 255, 255), -1)
                    cv2.putText(display_frame, f"Жест: {current_letter}", (start_x, start_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                    display_frame[start_y:start_y + h, start_x:start_x + w] = img

                # Прогресс
                progress = (read_index + 1) / len(read_letters)
                cv2.rectangle(display_frame, (50, frame_height - 100), (frame_width - 50, frame_height - 80),
                              (200, 200, 200), -1)
                cv2.rectangle(display_frame, (50, frame_height - 100),
                              (50 + int((frame_width - 100) * progress), frame_height - 80), (0, 200, 0), -1)
                cv2.putText(display_frame, f"{read_index + 1}/{len(read_letters)}",
                            (frame_width // 2 - 100, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255),
                            3)
            else:
                current_mode = MODE_INPUT
                read_letters = []
                read_index = 0
                read_timer = 0
        else:
            current_mode = MODE_INPUT

        frame = display_frame

    # === РЕЖИМ ПАУЗЫ ===
    elif current_mode == MODE_PAUSE:
        pass

    # === Отрисовка интерфейса ===
    x = LEFT_MARGIN
    y = TOP_MARGIN

    mode_text = {
        MODE_INPUT: "Режим: Жесты → Текст (1)",
        MODE_READ: "Режим: Текст → Жесты (2)",
        MODE_PAUSE: "Режим: Пауза (3)"
    }[current_mode]
    mode_color = {
        MODE_INPUT: (0, 255, 0),
        MODE_READ: (0, 255, 255),
        MODE_PAUSE: (0, 100, 255)
    }[current_mode]

    frame = draw_text_cyr(frame, mode_text, x, y, small_font, mode_color)
    y += 60

    if current_mode == MODE_INPUT:
        frame = draw_text_cyr(frame, "Распознано:", x, y, small_font, (0, 255, 0))
        y += 60
        if pred_text:
            frame = draw_text_cyr(frame, pred_text, x, y, font, (0, 255, 0))
        else:
            frame = draw_text_cyr(frame, "Покажите жест", x, y, font, (200, 200, 200))
        y += 90

    frame = draw_text_cyr(frame, "Текст:", x, y, small_font, (255, 255, 255))
    y += 50

    if text.strip():
        frame = draw_wrapped_text(frame, text, x, y, font, (0, 0, 0), max_text_width)
    else:
        frame = draw_text_cyr(frame, "(начните дактилировать)", x, y, font, (100, 100, 100))

    status = f"Задержка: {CONFIRM_FRAMES} кадров | 1/2/3 - режимы | + / - задержка"
    frame = draw_text_cyr(frame, status, x, frame_height - 50, small_font, (255, 255, 0))

    cv2.imshow("Дактиль РЖЯ - Реальное время", frame)

    # === Клавиши ===
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('1'):
        current_mode = MODE_INPUT
    elif key == ord('2'):
        if text.strip():
            fixed_background = frame.copy()
            read_letters = [c for c in text.upper() if c in CLASSES]
            read_index = 0
            read_timer = 0
            current_mode = MODE_READ
    elif key == ord('3'):
        current_mode = MODE_PAUSE
    elif key == 8 and current_mode == MODE_INPUT:
        if text:
            text = text[:-1]
            confirmed_letter = None
    elif key == 32 and current_mode == MODE_INPUT:
        text += " "
    elif key == 13 and current_mode == MODE_INPUT:
        text += "\n"
    elif key == ord('c') or key == ord('C'):
        text = ""
        confirmed_letter = None
    elif key == ord('+') or key == ord('='):
        CONFIRM_FRAMES = min(40, CONFIRM_FRAMES + 2)
    elif key == ord('-'):
        CONFIRM_FRAMES = max(5, CONFIRM_FRAMES - 2)

cap.release()
cv2.destroyAllWindows()
"""
"""
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import sys

# ================== НАСТРОЙКИ ==================
APP_MAIN = "mvp_model.py"      # Основной переводчик
APP_TEST = "test_mode2.py"     # Тестовый режим
# ===============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXEC = sys.executable   # Текущий интерпретатор Python


def run_python_app(script_name):
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.exists(script_path):
        messagebox.showerror(
            "Ошибка",
            f"Файл не найден:\n{script_name}"
        )
        return

    try:
        subprocess.Popen([PYTHON_EXEC, script_path])
    except Exception as e:
        messagebox.showerror(
            "Ошибка запуска",
            str(e)
        )


# ================== UI ==================
root = tk.Tk()
root.title("Переводчик РЖЯ")
root.geometry("520x320")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("default")

# Заголовок
header = ttk.Label(
    root,
    text="Переводчик русского жестового языка",
    font=("Segoe UI", 14, "bold")
)
header.pack(pady=(15, 5))

subtitle = ttk.Label(
    root,
    text="Система распознавания жестов и перевода в текст",
    font=("Segoe UI", 10)
)
subtitle.pack(pady=(0, 10))

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=15, pady=10)

# ================== Вкладка 1 ==================
tab_main = ttk.Frame(notebook)
notebook.add(tab_main, text="🖐 Переводчик")

ttk.Label(
    tab_main,
    text="Основной режим перевода РЖЯ",
    font=("Segoe UI", 11, "bold")
).pack(pady=(30, 10))

ttk.Label(
    tab_main,
    text="Распознавание жестов с камеры\nи преобразование в текст",
    font=("Segoe UI", 10),
    justify="center"
).pack(pady=(0, 20))

ttk.Button(
    tab_main,
    text="Запустить переводчик",
    width=35,
    command=lambda: run_python_app(APP_MAIN)
).pack()

# ================== Вкладка 2 ==================
tab_test = ttk.Frame(notebook)
notebook.add(tab_test, text="🧪 Тестирование")

ttk.Label(
    tab_test,
    text="Тестовый / обучающий режим",
    font=("Segoe UI", 11, "bold")
).pack(pady=(30, 10))

ttk.Label(
    tab_test,
    text="Проверка модели, отладка\nи тестирование жестов",
    font=("Segoe UI", 10),
    justify="center"
).pack(pady=(0, 20))

ttk.Button(
    tab_test,
    text="Запустить тестовый режим",
    width=35,
    command=lambda: run_python_app(APP_TEST)
).pack()

# ================== FOOTER ==================
footer = ttk.Label(
    root,
    text="© Проект РЖЯ • MVP",
    font=("Segoe UI", 9)
)
footer.pack(pady=(5, 10))

root.mainloop()
"""
import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import runpy
import threading

# ================== НАСТРОЙКИ ==================
APP_MAIN = "mvp_model.py"
APP_TEST = "test_mode2.py"
ICON_FILE = "icon.ico"


def get_resource_path(relative_path):
    """Находит путь к файлу внутри EXE или при разработке"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def launch_script(script_name, button):
    """Запуск скрипта в отдельном потоке с отключением кнопки"""
    script_path = get_resource_path(script_name)

    if not os.path.exists(script_path):
        messagebox.showerror(
            "Ошибка",
            f"Файл не найден: {script_name}\n\n"
            f"Путь: {script_path}\n\n"
            "Проверьте правильность сборки PyInstaller и наличие всех файлов."
        )
        return

    # Отключаем кнопку, чтобы избежать повторных запусков
    button['state'] = 'disabled'

    def run():
        try:
            runpy.run_path(script_path, run_name="__main__")
        except Exception as e:
            messagebox.showerror(
                "Ошибка запуска",
                f"Не удалось запустить {script_name}\n\n"
                f"Ошибка: {str(e)}"
            )
        finally:
            # Возвращаем кнопку в нормальное состояние (на случай ошибки)
            root.after(0, lambda: button.config(state='normal'))

    # Запуск в отдельном потоке
    threading.Thread(target=run, daemon=True).start()


# ================== UI ==================
root = tk.Tk()
root.title("Переводчик РЖЯ")
root.geometry("520x320")
root.resizable(False, False)

# Центрирование окна на экране
root.eval('tk::PlaceWindow . center')

# Иконка окна
icon_path = get_resource_path(ICON_FILE)
if os.path.exists(icon_path):
    try:
        root.iconbitmap(icon_path)  # Для .ico на Windows
    except:
        pass  # Если не сработает — просто пропустим

style = ttk.Style()
style.theme_use("default")

# Заголовок
ttk.Label(root, text="Переводчик русского жестового языка",
          font=("Segoe UI", 14, "bold")).pack(pady=(15, 5))
ttk.Label(root, text="Система распознавания жестов и перевода в текст",
          font=("Segoe UI", 10)).pack(pady=(0, 10))

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both", padx=15, pady=10)

# Вкладка 1 — Переводчик
tab_main = ttk.Frame(notebook)
notebook.add(tab_main, text="🖐 Переводчик")

ttk.Label(tab_main, text="Основной режим перевода РЖЯ",
          font=("Segoe UI", 11, "bold")).pack(pady=(30, 10))
ttk.Label(tab_main, text="Распознавание жестов с камеры\nи преобразование в текст",
          font=("Segoe UI", 10), justify="center").pack(pady=(0, 20))

btn_main = ttk.Button(tab_main, text="🚀 Запустить переводчик", width=35)
btn_main.pack(pady=(0, 20))
btn_main.configure(command=lambda: launch_script(APP_MAIN, btn_main))

# Вкладка 2 — Тестирование
tab_test = ttk.Frame(notebook)
notebook.add(tab_test, text="🧪 Тестирование")

ttk.Label(tab_test, text="Тестовый / обучающий режим",
          font=("Segoe UI", 11, "bold")).pack(pady=(30, 10))
ttk.Label(tab_test, text="Проверка модели, отладка\nи тестирование жестов",
          font=("Segoe UI", 10), justify="center").pack(pady=(0, 20))

btn_test = ttk.Button(tab_test, text="🎯 Запустить тестовый режим", width=35)
btn_test.pack(pady=(0, 20))
btn_test.configure(command=lambda: launch_script(APP_TEST, btn_test))

# Футер
footer_frame = ttk.Frame(root)
footer_frame.pack(pady=(5, 10))

ttk.Label(footer_frame, text="© Проект РЖЯ • MVP 2026", font=("Segoe UI", 9)).pack(side="left", padx=20)

ttk.Button(footer_frame, text="✖ Выход", command=root.quit).pack(side="right", padx=20)

root.mainloop()
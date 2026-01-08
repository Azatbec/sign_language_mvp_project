"""
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import SignTransformer  # Теперь будет работать после переименования

# ======== CONFIG ========
MODEL_PATH = "bukva/best_model_fixed.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
BASE_DIM = 63
USE_MOTION = True
INPUT_DIM = BASE_DIM * 3 if USE_MOTION else BASE_DIM

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Загрузка модели
model = SignTransformer(input_dim=189).to(DEVICE)  # d_model=256 по умолчанию
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Пре-процессинг
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

# Буфер
buffer = []
pred_history = []
SMOOTH_WINDOW = 7

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)
    pred_text = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        # Скелет руки
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

                pred_history.append(pred_id)
                if len(pred_history) > SMOOTH_WINDOW:
                    pred_history.pop(0)

                if len(pred_history) == SMOOTH_WINDOW:
                    final_id = max(set(pred_history), key=pred_history.count)
                    pred_text = CLASSES[final_id]

    else:
        buffer.clear()
        pred_history.clear()

    # Отображение
    if pred_text:
        cv2.putText(frame, pred_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
        cv2.putText(frame, "Распознано:", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    else:
        cv2.putText(frame, "Покажите жест", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (200, 200, 200), 5)

    cv2.imshow("Дактиль РЖЯ - Реальное время", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import SignTransformer

# === КИРИЛЛИЦА ДЛЯ OpenCV ===
from PIL import ImageFont, ImageDraw, Image

# Загрузи шрифт — положи arial.ttf рядом с файлом!
font = ImageFont.truetype("arial.ttf", 80)
small_font = ImageFont.truetype("arial.ttf", 50)

def draw_text_cyr(frame, text, x, y, font, color=(0,255,0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font, fill=color)
    frame = np.array(img_pil)
    return frame


# ======== CONFIG ========
MODEL_PATH = "bukva/best_model_fixed.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
BASE_DIM = 63
USE_MOTION = True
INPUT_DIM = BASE_DIM * 3 if USE_MOTION else BASE_DIM

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Load model
model = SignTransformer(input_dim=189).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Пре-процессинг
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


# Буферы
buffer = []
pred_history = []
SMOOTH_WINDOW = 7

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)
    pred_text = ""

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

                pred_history.append(pred_id)
                if len(pred_history) > SMOOTH_WINDOW:
                    pred_history.pop(0)

                if len(pred_history) == SMOOTH_WINDOW:
                    final_id = max(set(pred_history), key=pred_history.count)
                    pred_text = CLASSES[final_id]

    else:
        buffer.clear()
        pred_history.clear()

    # === КИРИЛЛИЦА НА ЭКРАНЕ ===
    if pred_text:
        frame = draw_text_cyr(frame, "Распознано:", 50, 20, small_font, (0,255,0))
        frame = draw_text_cyr(frame, pred_text, 50, 120, font, (0,255,0))
    else:
        frame = draw_text_cyr(frame, "Покажите жест", 50, 120, font, (200,200,200))

    cv2.imshow("Дактиль РЖЯ - Реальное время", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import SignTransformer

# === КИРИЛЛИЦА ДЛЯ OpenCV ===
from PIL import ImageFont, ImageDraw, Image

# Шрифты
font = ImageFont.truetype("arial.ttf", 60)  # Основной текст
small_font = ImageFont.truetype("arial.ttf", 40)  # Подписи


def draw_text_cyr(frame, text, x, y, font_obj, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font_obj, fill=color)
    return np.array(img_pil)


# Автоматический перенос текста по словам с границами
def draw_wrapped_text(frame, text, start_x, start_y, font_obj, color, max_width):
    if not text:
        return frame

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font_obj)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    line_height = font_obj.size + 15
    y = start_y
    for line in lines:
        draw.text((start_x, y), line, font=font_obj, fill=color)
        y += line_height

    return np.array(img_pil)


# ======== CONFIG ========
MODEL_PATH = "bukva/best_model_fixed.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
BASE_DIM = 63
USE_MOTION = True

CLASSES = ['Ё', 'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л',
           'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш',
           'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я']

# Load model
model = SignTransformer(input_dim=189).to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# MediaPipe
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Пре-процессинг
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


# Буферы
buffer = []
pred_history = []
SMOOTH_WINDOW = 7

# Накопленный текст и состояние
text = ""
last_pred = None  # Теперь None в начале

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Границы текста
LEFT_MARGIN = 50
RIGHT_MARGIN = 50
TOP_MARGIN = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    max_text_width = frame_width - LEFT_MARGIN - RIGHT_MARGIN

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)
    pred_text = ""

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

                pred_history.append(pred_id)
                if len(pred_history) > SMOOTH_WINDOW:
                    pred_history.pop(0)

                if len(pred_history) == SMOOTH_WINDOW:
                    final_id = max(set(pred_history), key=pred_history.count)
                    pred_text = CLASSES[final_id]

                    # === Логика добавления буквы под жестовый ввод ===
                    if pred_text != last_pred:  # Смена жеста
                        if last_pred is not None:  # Не первая буква
                            if pred_text == last_pred:
                                # Повтор той же буквы — добавляем без пробела (для "СС", "ЛЛ" и т.д.)
                                text += pred_text
                            else:
                                # Новая буква — добавляем с пробелом
                                text += " " + pred_text
                        else:
                            # Первая буква в сессии
                            text += pred_text
                        last_pred = pred_text

    else:
        buffer.clear()
        pred_history.clear()
        last_pred = None  # Сброс при исчезновении руки

    # === Отрисовка ===
    x = LEFT_MARGIN
    y = TOP_MARGIN

    # Текущий жест
    frame = draw_text_cyr(frame, "Распознано:", x, y, small_font, (0, 255, 0))
    y += 60
    if pred_text:
        frame = draw_text_cyr(frame, pred_text, x, y, font, (0, 255, 0))
    else:
        frame = draw_text_cyr(frame, "Покажите жест", x, y, font, (200, 200, 200))
    y += 90

    # Накопленный текст (чёрный, с переносом)
    frame = draw_text_cyr(frame, "Текст:", x, y, small_font, (255, 255, 255))
    y += 50

    cleaned_text = text.strip()
    if cleaned_text:
        frame = draw_wrapped_text(frame, cleaned_text, x, y, font, (0, 0, 0), max_text_width)
    else:
        frame = draw_text_cyr(frame, "(начните дактилировать)", x, y, font, (100, 100, 100))

    cv2.imshow("Дактиль РЖЯ - Реальное время", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC — выход
        break
    elif key == 8:  # Backspace — удаление
        text = text.rstrip(" ")  # Убираем лишние пробелы с конца
        if text:
            text = text[:-1]
    elif key == ord('c') or key == ord('C'):  # Очистка по клавише C
        text = ""
        last_pred = None

cap.release()
cv2.destroyAllWindows()
"""
"""
#45
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import SignTransformer

# === КИРИЛЛИЦА ДЛЯ OpenCV ===
from PIL import ImageFont, ImageDraw, Image

font = ImageFont.truetype("arial.ttf", 60)
small_font = ImageFont.truetype("arial.ttf", 40)

def draw_text_cyr(frame, text, x, y, font_obj, color=(0,255,0)):
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
state = torch.load(MODEL_PATH, map_location=DEVICE)
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


# === Переменные состояния ===
buffer = []
text = ""
current_pred = None        # Текущий распознанный жест (для отображения)
confirmed_letter = None    # Последняя добавленная буква
stable_frames = 0          # Сколько кадров подряд держится текущий жест

# Настраиваемая задержка подтверждения (можно менять на лету)
CONFIRM_FRAMES = 15  # Начальное значение: ~0.5 сек при 30 fps

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

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(frame_rgb)
    pred_text = ""

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

                # === Объединённая логика с условиями и циклом ===
                if pred_text == current_pred:
                    stable_frames += 1
                else:
                    current_pred = pred_text
                    stable_frames = 1

                # Проверка на подтверждение буквы
                if (stable_frames >= CONFIRM_FRAMES and
                    pred_text != confirmed_letter):
                    text += pred_text
                    confirmed_letter = pred_text
                    stable_frames = 0  # Сброс, чтобы не повторять при долгом удержании

    else:
        buffer.clear()
        current_pred = None
        confirmed_letter = None
        stable_frames = 0
        pred_text = ""

    # === Отрисовка ===
    x = LEFT_MARGIN
    y = TOP_MARGIN

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

    # Показ текущей задержки (для удобства настройки)
    frame = draw_text_cyr(frame, f"Задержка: {CONFIRM_FRAMES} кадров", x, frame_height - 50, small_font, (255, 255, 0))

    cv2.imshow("Дактиль РЖЯ - Реальное время", frame)

    # === Клавиши ===
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 8:  # Backspace
        if text:
            text = text[:-1]
            confirmed_letter = None
    elif key == 32:  # Пробел
        text += " "
    elif key == 13:  # Enter
        text += "\n"
    elif key == ord('c') or key == ord('C'):
        text = ""
        confirmed_letter = None
    elif key == ord('+') or key == ord('='):  # Увеличить задержку (медленнее)
        CONFIRM_FRAMES = min(40, CONFIRM_FRAMES + 2)
    elif key == ord('-'):  # Уменьшить задержку (быстрее)
        CONFIRM_FRAMES = max(5, CONFIRM_FRAMES - 2)

cap.release()
cv2.destroyAllWindows()
"""

"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F

# ===== CONFIG =====
H5_PATH = "bukva/processed_data/processed_dataset.h5"
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 33
SEQ_LEN = 32
USE_MOTION = True
D_MODEL = 512
NHEAD = 16
NUM_LAYERS = 8
DROPOUT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ===== Сильные аугментации для похожих букв =====
def strong_augment(seq):
    seq = seq.copy()

    # Шум
    seq += np.random.normal(0, 0.03, seq.shape)

    # Масштабирование
    scale = np.random.uniform(0.8, 1.2)
    seq *= scale

    # Поворот руки
    theta = np.random.uniform(-0.4, 0.4)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    xy = seq[:, :42].reshape(SEQ_LEN, 21, 2)
    xy = np.matmul(xy, rot)
    seq[:, :42] = xy.reshape(SEQ_LEN, 42)

    # Временное искажение
    alpha = np.random.uniform(0.8, 1.2)
    t = np.linspace(0, 1, SEQ_LEN)
    t_warped = np.power(t, alpha)
    new_idx = np.clip((t_warped * SEQ_LEN).astype(int), 0, SEQ_LEN - 1)
    seq = seq[new_idx]

    # Случайное скрытие кадров
    if random.random() < 0.4:
        mask_len = random.randint(2, 6)
        start = random.randint(0, SEQ_LEN - mask_len)
        seq[start:start + mask_len] = 0

    return seq.astype(np.float32)


# ===== Dataset =====
class BukvaDataset(Dataset):
    def __init__(self, h5_file, split="train", use_motion=USE_MOTION, augment=False):
        self.use_motion = use_motion
        self.augment = augment
        self.features = []
        self.labels = []

        with h5py.File(h5_file, 'r') as f:
            grp = f[split]
            for key in tqdm(grp.keys(), desc=f"Loading {split}"):
                if key.endswith("_features"):
                    base = key.replace("_features", "")
                    feats = grp[key][:]
                    label = int(grp[base + "_label"][()])
                    self.features.append(feats)
                    self.labels.append(label)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        if use_motion:
            motion_feats = []
            for seq in tqdm(self.features, desc="Adding motion"):
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                acc = np.diff(vel, axis=0, prepend=vel[:1])
                full = np.concatenate([seq, vel, acc], axis=1)
                motion_feats.append(full)
            self.features = np.array(motion_feats, dtype=np.float32)

        print(f"{split.upper()}: {len(self)} samples, dim {self.features.shape[-1]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        seq = self.features[idx].copy()
        if self.augment:
            seq = strong_augment(seq)
        return torch.from_numpy(seq), torch.tensor(self.labels[idx], dtype=torch.long)


# ===== Focal Loss =====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ===== Мощная модель =====
class SignTransformer(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # Улучшенный TCN
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, dilation=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(d_model, d_model, 3, padding=2, dilation=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Conv1d(d_model, d_model, 3, padding=4, dilation=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            dropout=DROPOUT, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ===== Обучение =====
def main():
    train_ds = BukvaDataset(H5_PATH, split="train", augment=True)
    val_ds = BukvaDataset(H5_PATH, split="test", augment=False)

    input_dim = train_ds.features.shape[-1]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SignTransformer(input_dim=input_dim).to(DEVICE)
    criterion = FocalLoss(gamma=2.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        preds, trues = [], []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.cpu().numpy())
        train_acc = accuracy_score(trues, preds)

        model.eval()
        val_preds, val_trues = [], []
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item()
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_trues.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_trues, val_preds)

        scheduler.step()

        print(
            f"Epoch {epoch:03d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "bukva/best_final_model.pth")
            print(f"  → Новый рекорд: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    print(f"Готово! Лучшая точность: {best_acc:.4f}")


if __name__ == "__main__":
    main()
"""
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast  # Новый API
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

# ===== CONFIG =====
H5_PATH = "bukva/processed_data/processed_dataset.h5"
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 33
SEQ_LEN = 32
USE_MOTION = True
D_MODEL = 512
NHEAD = 16
NUM_LAYERS = 8
DROPOUT = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ===== Сильные аугментации =====
def strong_augment(seq):
    seq = seq.copy().astype(np.float32)

    # Шум
    seq += np.random.normal(0, 0.04, seq.shape)

    # Масштаб
    seq *= np.random.uniform(0.8, 1.2)

    # Поворот
    theta = np.random.uniform(-0.45, 0.45)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    xy = seq[:, :42].reshape(SEQ_LEN, 21, 2)
    xy = np.matmul(xy, rot)
    seq[:, :42] = xy.reshape(SEQ_LEN, 42)

    # Временное искажение
    alpha = np.random.uniform(0.75, 1.25)
    t = np.linspace(0, 1, SEQ_LEN)
    t_warped = np.power(t, alpha)
    new_idx = np.clip(np.floor(t_warped * SEQ_LEN).astype(int), 0, SEQ_LEN - 1)
    seq = seq[new_idx]

    # Случайное скрытие кадров
    if random.random() < 0.5:
        mask_len = random.randint(3, 8)
        start = random.randint(0, SEQ_LEN - mask_len)
        seq[start:start + mask_len] = 0

    return seq


# ===== Dataset =====
class BukvaDataset(Dataset):
    def __init__(self, h5_file, split="train", use_motion=USE_MOTION, augment=False):
        self.use_motion = use_motion
        self.augment = augment
        self.features = []
        self.labels = []

        with h5py.File(h5_file, 'r') as f:
            grp = f[split]
            for key in tqdm(grp.keys(), desc=f"Loading {split}"):
                if key.endswith("_features"):
                    base = key.replace("_features", "")
                    feats = grp[key][:]
                    label = int(grp[base + "_label"][()])
                    self.features.append(feats)
                    self.labels.append(label)

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

        if use_motion:
            motion_feats = []
            for seq in tqdm(self.features, desc="Adding motion"):
                vel = np.diff(seq, axis=0, prepend=seq[:1])
                acc = np.diff(vel, axis=0, prepend=vel[:1])
                full = np.concatenate([seq, vel, acc], axis=1)
                motion_feats.append(full)
            self.features = np.array(motion_feats, dtype=np.float32)

        print(f"{split.upper()}: {len(self)} samples, dim {self.features.shape[-1]}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        seq = self.features[idx].copy()
        if self.augment:
            seq = strong_augment(seq)
        return torch.from_numpy(seq), torch.tensor(self.labels[idx], dtype=torch.long)


# ===== Focal Loss =====
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


# ===== Улучшенная модель =====
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.25):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out[..., :x.size(-1)]
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.res_proj:
            residual = self.res_proj(residual)
        return out + residual


class SignTransformer(nn.Module):
    def __init__(self, input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.tcn = nn.Sequential(
            TCNBlock(d_model, d_model, dropout=DROPOUT),
            TCNBlock(d_model, d_model, dilation=2, dropout=DROPOUT),
            TCNBlock(d_model, d_model, dilation=4, dropout=DROPOUT),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.4),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


# ===== Обучение =====
def main():
    train_ds = BukvaDataset(H5_PATH, split="train", augment=True)
    val_ds = BukvaDataset(H5_PATH, split="test", augment=False)

    input_dim = train_ds.features.shape[-1]
    print(f"Input dim: {input_dim}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = SignTransformer(input_dim=input_dim).to(DEVICE)
    criterion = FocalLoss(gamma=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()  # Правильно для PyTorch 2+
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with autocast(device_type='cuda' if DEVICE == "cuda" else 'cpu'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with autocast(device_type='cuda' if DEVICE == "cuda" else 'cpu'):
                    logits = model(x)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_trues.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_trues, val_preds)

        scheduler.step()

        print(f"Epoch {epoch:03d} | Train Loss: {total_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "bukva/best_final_model.pth")
            print(f"  → Новый рекорд: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    print(f"Обучение завершено. Лучшая точность: {best_acc:.4f}")


if __name__ == "__main__":
    main()
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import sys
from model import SignTransformer

# ===================== PYINSTALLER PATH =====================
def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ===================== КИРИЛЛИЦА =====================
from PIL import ImageFont, ImageDraw, Image

try:
    font = ImageFont.truetype(resource_path("arial.ttf"), 60)
    small_font = ImageFont.truetype(resource_path("arial.ttf"), 40)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()

def draw_text_cyr(frame, text, x, y, font_obj, color=(0,255,0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), text, font=font_obj, fill=color)
    return np.array(img_pil)

def draw_wrapped_text(frame, text, start_x, start_y, font_obj, color, max_width):
    if not text:
        return frame

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    y = start_y
    line_height = font_obj.size + 15

    for line in text.split('\n'):
        words = line.split()
        current = ""

        for word in words:
            test = current + (" " if current else "") + word
            w = draw.textlength(test, font=font_obj)

            if w <= max_width:
                current = test
            else:
                draw.text((start_x, y), current, font=font_obj, fill=color)
                y += line_height
                current = word

        if current:
            draw.text((start_x, y), current, font=font_obj, fill=color)
            y += line_height + 10

    return np.array(img_pil)

# ===================== CONFIG =====================
MODEL_PATH = resource_path("bukva/best_model_fixed.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 32
USE_MOTION = True

CLASSES = [
    'Ё','А','Б','В','Г','Д','Е','Ж','З','И','Й','К','Л','М','Н','О','П',
    'Р','С','Т','У','Ф','Х','Ц','Ч','Ш','Щ','Ъ','Ы','Ь','Э','Ю','Я'
]

# ===================== MODEL =====================
model = SignTransformer(input_dim=189).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# ===================== MEDIAPIPE =====================
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ===================== FEATURES =====================
def normalize_landmarks(landmarks):
    if not landmarks:
        return np.zeros(63, dtype=np.float32)

    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    arr -= arr[0]
    m = np.linalg.norm(arr, axis=1).max()
    return (arr / m).flatten() if m > 0 else arr.flatten()

def add_motion_features(seq):
    vel = np.diff(seq, axis=0, prepend=seq[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    return np.concatenate([seq, vel, acc], axis=1)

def per_sample_normalize(x):
    mx = np.max(np.abs(x))
    return x / mx if mx > 0 else x

# ===================== STATE =====================
buffer = []
text = ""
current_pred = None
confirmed_letter = None
stable_frames = 0
CONFIRM_FRAMES = 15

# ===================== CAMERA =====================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

LEFT = 50
TOP = 30

# ===================== LOOP =====================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    max_w = w - LEFT * 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)
    pred_text = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand, mp.solutions.hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        buffer.append(normalize_landmarks(hand.landmark))
        if len(buffer) > SEQ_LEN:
            buffer.pop(0)

        if len(buffer) == SEQ_LEN:
            seq = np.array(buffer, dtype=np.float32)
            if USE_MOTION:
                seq = add_motion_features(seq)
            seq = per_sample_normalize(seq)[None, ...]

            with torch.no_grad():
                logits = model(torch.from_numpy(seq).to(DEVICE))
                pred_text = CLASSES[logits.argmax(1).item()]

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

    y = TOP
    frame = draw_text_cyr(frame, "Распознано:", LEFT, y, small_font)
    y += 60
    frame = draw_text_cyr(frame, pred_text or "Покажите жест", LEFT, y, font)
    y += 100
    frame = draw_text_cyr(frame, "Текст:", LEFT, y, small_font, (255,255,255))
    y += 50

    frame = draw_wrapped_text(
        frame,
        text or "(начните дактилировать)",
        LEFT, y, font,
        (0,0,0), max_w
    )

    frame = draw_text_cyr(
        frame,
        f"Задержка: {CONFIRM_FRAMES} кадров",
        LEFT, h - 50, small_font, (255,255,0)
    )

    cv2.imshow("Дактиль РЖЯ", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 8 and text:
        text = text[:-1]
        confirmed_letter = None
    elif key == 32:
        text += " "
    elif key == 13:
        text += "\n"
    elif key in (ord('+'), ord('=')):
        CONFIRM_FRAMES = min(40, CONFIRM_FRAMES + 2)
    elif key == ord('-'):
        CONFIRM_FRAMES = max(5, CONFIRM_FRAMES - 2)
    elif key in (ord('c'), ord('C')):
        text = ""
        confirmed_letter = None

cap.release()
cv2.destroyAllWindows()

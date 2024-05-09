import joblib
from PIL import Image, ImageFilter
import numpy as np


def load_model(model_path):
    trained_model = joblib.load(model_path)
    return trained_model


def recognize_digit(image_path, model):
    # Загрузка изображения и преобразование в градации серого
    image = Image.open(image_path).convert('L')

    # Применение фильтра Гаусса для сглаживания и уменьшения шума
    image = image.filter(ImageFilter.GaussianBlur(1))

    # Использование пороговой обработки для улучшения контраста
    threshold = 100
    image = image.point(lambda p: p > threshold and 255)

    # Изменение размера изображения и преобразование в массив
    img_resized = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img_resized).reshape(1, -1)

    # Инвертирование пикселей изображения
    img_inv = 255 - img_array

    # Нормализация изображения
    img_norm = img_inv / 255.0

    digit_predicted = model.predict(img_norm)
    return digit_predicted[0]

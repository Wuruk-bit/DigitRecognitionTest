import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from utils import recognize_digit, load_model


class DigitRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = load_model('model.joblib')  # Загрузка модели один раз при инициализации
        self.filename = None
        self.initUI()

    def initUI(self):
        # Кнопка для загрузки изображения
        self.btn_load = QPushButton('Загрузить изображение', self)
        self.btn_load.clicked.connect(self.openFileNameDialog)

        # Кнопка для анализа изображения
        self.btn_analyze = QPushButton('Анализ цифры на изображении', self)
        self.btn_analyze.clicked.connect(self.analyzeDigit)

        self.image_label = QLabel(self)
        self.result_label = QLabel('Результат распознавания будет здесь', self)

        layout = QVBoxLayout()
        layout.addWidget(self.btn_load)
        layout.addWidget(self.image_label)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.setWindowTitle('Распознавание цифр')
        self.setGeometry(300, 300, 350, 350)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "",
                                                  "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if filename:
            self.filename = filename
            pixmap = QPixmap(self.filename)
            self.image_label.setPixmap(pixmap.scaled(128, 128))  # Отображение изображения

    def analyzeDigit(self):
        if self.filename:
            predicted = recognize_digit(self.filename, self.model)
            self.result_label.setText(f'Результат распознавания: {predicted}')
        else:
            self.result_label.setText('Изображение не выбрано')


def main():
    app = QApplication(sys.argv)
    ex = DigitRecognizerApp()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

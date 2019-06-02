import numpy as np
import yaml

from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QVBoxLayout, QApplication, QSlider, QLabel, QHBoxLayout, QScrollArea, QGroupBox
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image

class ProcessingWindow(QMainWindow):
    SETTING_FILENAME = 'settings/trackbar_settings.yaml'
    SCALING_FACTOR = 0.4

    def __init__(self, processing):
        super(ProcessingWindow, self).__init__()
        self.processing = processing
        
        self.settings = yaml.load(open(self.SETTING_FILENAME))[0]
        self.processing.update_settings(self.settings)

        self.setup_sliders()
        self.setup_images()

        self.central_widget = QWidget()

        mygroupbox = QGroupBox('Settings')
        mygroupbox.setLayout(self.slider_layout)
        scroll = QScrollArea()
        scroll.setWidget(mygroupbox)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(200)


        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.addLayout(self.image_layout)
        #self.main_layout.addLayout(self.slider_layout)
        #self.main_layout.addWidget(mygroupbox)
        self.main_layout.addWidget(scroll)
        self.setCentralWidget(self.central_widget)


        self.update_timer = QTimer()
        #TODO: Set correct function
        self.update_timer.timeout.connect(self.update_images)
        self.update_timer.start(50)


    def setup_sliders(self):
        self.average_slider = LabeledSlider('Avg_filt', 'avg', self.settings['avg'], 1, 10, lambda value:self.update_slider(value, 'avg'))
        self.gauss_slider = LabeledSlider('Gauss_filt', 'gauss', self.settings['gauss'], 1, 10, lambda value:self.update_slider(value, 'gauss'), ['odd'])
        self.median_slider = LabeledSlider('Med_filt', 'med', self.settings['med'], 1, 5, lambda value:self.update_slider(value, 'med'), ['odd'])

        self.h_low_slider = LabeledSlider('H_low', 'hl', self.settings['hl'], 0, 255, lambda value:self.update_slider(value, 'hl'))
        self.h_high_slider = LabeledSlider('H_high', 'hh', self.settings['hh'], 0, 255, lambda value:self.update_slider(value, 'hh'))
        self.s_low_slider = LabeledSlider('S_low', 'sl', self.settings['sl'], 0, 255, lambda value:self.update_slider(value, 'sl'))
        self.s_high_slider = LabeledSlider('S_high', 'sh', self.settings['sh'], 0, 255, lambda value:self.update_slider(value, 'sh'))
        self.v_low_slider = LabeledSlider('V_low', 'vl', self.settings['vl'], 0, 255, lambda value:self.update_slider(value, 'vl'))
        self.v_high_slider = LabeledSlider('V_high', 'vh', self.settings['vh'], 0, 255, lambda value:self.update_slider(value, 'vh'))

        self.erode_slider = LabeledSlider('Erode', 'erode', self.settings['erode'], 0, 10, lambda value:self.update_slider(value, 'erode'))
        self.dilate_slider = LabeledSlider('Dilate', 'dilate', self.settings['dilate'], 0, 10, lambda value:self.update_slider(value, 'dilate'))
        self.open_slider = LabeledSlider('Opening', 'open', self.settings['open'], 1, 10, lambda value:self.update_slider(value, 'open'))
        self.close_slider = LabeledSlider('Closing', 'close', self.settings['close'], 1, 10, lambda value:self.update_slider(value, 'close'))
        self.skel_slider = LabeledSlider('Skel', 'skel', self.settings['skel'], 0, 100, lambda value:self.update_slider(value, 'skel'))

        self.slider_layout = QVBoxLayout()
        self.slider_layout.addLayout(self.average_slider.layout)
        self.slider_layout.addLayout(self.gauss_slider.layout)
        self.slider_layout.addLayout(self.median_slider.layout)
        self.slider_layout.addLayout(self.h_low_slider.layout)
        self.slider_layout.addLayout(self.h_high_slider.layout)
        self.slider_layout.addLayout(self.s_low_slider.layout)
        self.slider_layout.addLayout(self.s_high_slider.layout)
        self.slider_layout.addLayout(self.v_low_slider.layout)
        self.slider_layout.addLayout(self.v_high_slider.layout)
        self.slider_layout.addLayout(self.erode_slider.layout)
        self.slider_layout.addLayout(self.dilate_slider.layout)
        self.slider_layout.addLayout(self.open_slider.layout)
        self.slider_layout.addLayout(self.close_slider.layout)
        self.slider_layout.addLayout(self.skel_slider.layout)


    def setup_images(self):
        self.clean_image = QLabel()
        self.hsv_image = QLabel()
        self.average_image = QLabel()
        self.gauss_image = QLabel()
        self.median_image = QLabel()
        self.erode_image = QLabel()
        self.dilate_image = QLabel()
        self.open_image = QLabel()
        self.close_image = QLabel()
        self.skel_image = QLabel()


        self.image_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(self.clean_image)
        row1.addWidget(self.average_image)
        row1.addWidget(self.gauss_image)

        row2 = QHBoxLayout()
        row2.addWidget(self.median_image)
        row2.addWidget(self.hsv_image)
        row2.addWidget(self.erode_image)

        row3 = QHBoxLayout()
        row3.addWidget(self.dilate_image)
        row3.addWidget(self.open_image)
        row3.addWidget(self.close_image)
        row3.addWidget(self.skel_image)


        self.image_layout.addLayout(row1)
        self.image_layout.addLayout(row2)
        self.image_layout.addLayout(row3)

    def update_slider(self, value, setting):
        self.settings[setting] = value
        #self.update_settings()

    def update_settings(self):
        self.processing.update_settings(self.settings)

    def update_images(self):
        self.update_settings()
        self.frame = self.processing.get_frame()
        if self.frame is None:
            return
        self.clean_image.setPixmap(self.convert(self.frame))

        average_frame = self.processing.average_filter(self.frame)
        self.average_image.setPixmap(self.convert(average_frame))

        gauss_frame = self.processing.gaussian_filter(average_frame)
        self.gauss_image.setPixmap(self.convert(gauss_frame))

        median_frame = self.processing.median_filter(gauss_frame)
        self.median_image.setPixmap(self.convert(median_frame))

        hsv_frame = self.processing.hsv_color_filter(median_frame)
        self.hsv_image.setPixmap(self.convert(hsv_frame))

        erode_frame = self.processing.erode(hsv_frame)
        self.erode_image.setPixmap(self.convert(erode_frame))

        dilate_frame = self.processing.dilate(erode_frame)
        self.dilate_image.setPixmap(self.convert(dilate_frame))

        open_frame = self.processing.opening(dilate_frame)
        self.open_image.setPixmap(self.convert(open_frame))

        close_frame = self.processing.closing(open_frame)
        self.close_image.setPixmap(self.convert(close_frame))

        skel_frame = self.processing.skeletonize(close_frame)
        self.skel_image.setPixmap(self.convert(skel_frame))


    def convert(self, frame):
        if len(frame.shape) == 2:
            pix = self.convert_greyscale(frame)
        else:
            pix = self.convert_color(frame)
        w = int(pix.width() * self.SCALING_FACTOR)
        h = int(pix.height() * self.SCALING_FACTOR)
        return pix.scaled(w, h)
        
    def convert_greyscale(self, frame):
        height, width = frame.shape
        bytesPerLine = 1 * width

        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        image = image.rgbSwapped()
        return QPixmap.fromImage(image)

    def convert_color(self, frame):
        height, width, colors = frame.shape
        bytesPerLine = 3 * width

        image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        image = image.rgbSwapped()
        return QPixmap.fromImage(image)

    def keyPressEvent(self, event):
     key = event.key()
     #print(key)

     if key == 83:
        self.save_settings()

    def save_settings(self):
        with open(self.SETTING_FILENAME, 'w') as f:
            print(yaml.dump(self.settings, default_flow_style=False))
            yaml.dump([self.settings], f)

class LabeledSlider():
    def __init__(self, label_text, key, initial_val, min_val, max_val, func, args = []):
        self.label_text = label_text
        self.key = key
        self.func = func
        self.args = args

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(initial_val)
        self.slider.valueChanged.connect(self.update_slider)
        #self.slider.setFixedWidth(750)

        self.label = QLabel()
        self.set_label(initial_val)
        self.label.setFixedWidth(100)

        self.layout = QHBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

    def update_slider(self, value):
        if 'odd' in self.args:
            value = value * 2 - 1
        self.set_label(value)
        self.func(value)

    def set_label(self, value):
        self.label.setText(str(self.label_text) + ': ' + str(value))



if __name__ == '__main__':
    app = QApplication([])
    window = StartWindow()
    window.show()
    app.exit(app.exec_())

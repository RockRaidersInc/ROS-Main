from view import ProcessingWindow
from model import Processing

from PyQt5.QtWidgets import QApplication

def main():
    source_type = 'cap'
    source = ''
    #source_type = 'ros'
    #source = '/zed_node/left/image_rect_color'
    p = Processing(source_type, source)


    app = QApplication([])
    window = ProcessingWindow(p)
    window.show()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()

import sys
from view import ProcessingWindow
from model import Processing

from PyQt5.QtWidgets import QApplication


def main():
    source_type = str(sys.argv[1])
    source = str(sys.argv[2]) # '/zed_node/left/image_rect_color'
    p = Processing(source_type, source)

    app = QApplication([])
    window = ProcessingWindow(p)
    window.show()
    app.exit(app.exec_())

if __name__ == '__main__':
    main()

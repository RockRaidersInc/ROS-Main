#!/usr/bin/env python

import rospy
import sys
from utils.trackbar_view import ProcessingWindow
from utils.trackbar_model import Processing

from PyQt5.QtWidgets import QApplication


def main():
    rospy.init_node('trackbar')

    source_type = rospy.get_param('source_type')
    source = rospy.get_param('source')
    p = Processing(source_type, source)

    app = QApplication([])
    window = ProcessingWindow(p)
    window.show()
    app.exit(app.exec_())

    sys.exit()

if __name__ == '__main__':
    main()

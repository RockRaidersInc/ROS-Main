#!/usr/bin/env python

import os
import rospkg
import rospy
import threading

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QLabel, QVBoxLayout

from std_msgs.msg import Float64
from topic_tools.srv import *

class BatteryMonitoringPlugin(Plugin):

    def voltage_callback(self, data):
        self._voltage_label.setText('Voltage: ' + str(data.data) + ' V')
    def current_callback(self, data):
        self._current_label.setText('Current: ' + str(data.data) + ' A')
    def current_drawn_callback(self, data):
        self._current_drawn_label.setText('Current Drawn: ' + str(data.data) + ' Ah')


    def __init__(self, context):
        super(BatteryMonitoringPlugin, self).__init__(context)
        #return
        # Give QObjects reasonable names
        self.setObjectName('BatteryMonitoringPlugin')
        rp = rospkg.RosPack()

        # Process standalone plugin command-line arguments
        #from argparse import ArgumentParser
        #parser = ArgumentParser()
        ## Add argument(s) to the parser.
        #parser.add_argument("-q", "--quiet", action="store_true",
        #              dest="quiet",
        #              help="Put plugin in silent mode")
        #args, unknowns = parser.parse_known_args(context.argv())
        #if not args.quiet:
        #    print 'arguments: ', args
        #    print 'unknowns: ', unknowns

        # Create QWidget
        self._widget = QWidget()


        self._container = QWidget()
        self._layout    = QVBoxLayout()
        self._container.setLayout(self._layout)

        self._voltage_label = QLabel('Voltage: ')
        self._current_label = QLabel('Current: ')
        self._current_drawn_label = QLabel('Current Drawn: ')

        self._layout.addWidget(self._voltage_label)
        self._layout.addWidget(self._current_label)
        self._layout.addWidget(self._current_drawn_label)

        context.add_widget(self._container)


        # Get path to UI file which is a sibling of this file
        # in this example the .ui and .py file are in the same folder
        #ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelection.ui')
        #ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelectionSimple.ui')
        # Extend the widget with all attributes and children from UI file
        #loadUi(ui_file, self._widget, {})
        # Give QObjects reasonable names
        self._widget.setObjectName('Battery Monitoring')
        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane.
        #if context.serial_number() > 1:
            #self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))


        self.running = True
        rate = rospy.Rate(1)
        def run():
            rospy.Subscriber("bms/voltage", Float64, self.voltage_callback)
            rospy.Subscriber("bms/current", Float64, self.current_callback)
            rospy.Subscriber("bms/current_drawn", Float64, self.current_drawn_callback)
            while not self.running:
                rate.sleep()

        self.run_thread = threading.Thread(target=run)
        self.run_thread.start()
            

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        self.running = False
        self.found_service = True

    #def save_settings(self, plugin_settings, instance_settings):
    #    # TODO save intrinsic configuration, usually using:
    #    # instance_settings.set_value(k, v)
    #    pass

    #def restore_settings(self, plugin_settings, instance_settings):
    #    # TODO restore intrinsic configuration, usually using:
    #    # v = instance_settings.value(k)
    #    pass

    #def trigger_configuration(self):
        # Comment in to signal that the plugin has a way to configure
        # This will enable a setting button (gear icon) in each dock widget title bar
        # Usually used to open a modal configuration dialog

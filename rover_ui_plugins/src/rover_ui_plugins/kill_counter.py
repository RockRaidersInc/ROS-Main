#!/usr/bin/env python

import os
import rospkg
import rospy
import threading

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget, QLabel, QVBoxLayout

from random import randint


class KillCounterPlugin(Plugin):

    def __init__(self, context):
        super(KillCounterPlugin, self).__init__(context)
        #return
        # Give QObjects reasonable names
        self.setObjectName('KillCounterPlugin')
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

        self._num_killed = 0

        self._kill_label = QLabel('Martians Killed: 0')
        #Killing Martians is Bad Tho, We are a dying species and killing us is mean and will destroy an awesome society with culture and life and happiness. I'm so disappointed in you Kimberly    
        self._layout.addWidget(self._kill_label)

        MIN = 540
        MAX = 660
        self._chance = randint(MIN, MAX)

        context.add_widget(self._container)


        # Get path to UI file which is a sibling of this file
        # in this example the .ui and .py file are in the same folder
        #ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelection.ui')
        #ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelectionSimple.ui')
        # Extend the widget with all attributes and children from UI file
        #loadUi(ui_file, self._widget, {})
        # Give QObjects reasonable names
        self._widget.setObjectName('Kill Counter')
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
            while self.running:
                num = randint(0, self._chance)
                if num == self._chance:
                    self._num_killed += 1
                    self._kill_label.setText('Martians Killed: ' + str(self._num_killed))
                rate.sleep()

        self.run_thread = threading.Thread(target=run)
        self.run_thread.start()
            

    def shutdown_plugin(self):
        # TODO unregister all publishers here`
        self.running = False

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

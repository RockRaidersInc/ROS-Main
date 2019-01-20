#!/usr/bin/env python

import os
import rospkg
import rospy
import threading

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtWidgets import QWidget

from std_msgs.msg import String
from topic_tools.srv import *

class VideoSelectionPlugin(Plugin):

    def _pb1_clicked(self, checked):
        if self.found_service:
            self.mux(self.channels[0])

    def _pb2_clicked(self, checked):
        if self.found_service:
            self.mux(self.channels[1])


    def _pb3_clicked(self, checked):
        if self.found_service:
            self.mux(self.channels[2])


    def _pb4_clicked(self, checked):
        if self.found_service:
            self.mux(self.channels[3])



    def __init__(self, context):
        super(VideoSelectionPlugin, self).__init__(context)
        #return
        # Give QObjects reasonable names
        self.setObjectName('VideoSelectionPlugin')
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
        # Get path to UI file which is a sibling of this file
        # in this example the .ui and .py file are in the same folder
        #ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelection.ui')
        ui_file = os.path.join(rp.get_path('rover_ui_plugins'), 'resource', 'CameraSelectionSimple.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(ui_file, self._widget, {})
        # Give QObjects reasonable names
        self._widget.setObjectName('CameraSelection')
        # Show _widget.windowTitle on left-top of each plugin (when 
        # it's set in _widget). This is useful when you open multiple 
        # plugins at once. Also if you open multiple instances of your 
        # plugin at once, these lines add number to make it easy to 
        # tell from pane to pane.
        #if context.serial_number() > 1:
            #self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
        # Add widget to the user interface
        context.add_widget(self._widget)

        #self.pub = rospy.Publisher('image_mux/sel_image', String, queue_size=1)


        self.sub_num = 0
        self.channels = ['cam1', 'cam2', 'cam3', 'cam4']

        self._widget.pb1.clicked[bool].connect(self._pb1_clicked)
        self._widget.pb2.clicked[bool].connect(self._pb2_clicked)
        self._widget.pb3.clicked[bool].connect(self._pb3_clicked)
        self._widget.pb4.clicked[bool].connect(self._pb4_clicked)



        rate = rospy.Rate(10)
        self.running = True
        self.found_service = False
        def run():
            while (self.running):
                try:
                    rospy.wait_for_service('video_mux/select', 1)
                except:
                    pass

                
            self.found_service = True
            self.running = True
            self.mux = rospy.ServiceProxy('video_mux/select', MuxSelect)
            while not self.running:
                #self.pub.publish(self.channels[self.sub_num])
                rate.sleep()
                #print(self.sub_num)

        self.run_thread = threading.Thread(target=run)
        self.run_thread.start()
            

    def shutdown_plugin(self):
        # TODO unregister all publishers here
        self.running = False
        self.found_service = True
        #self.pub.unregister()

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

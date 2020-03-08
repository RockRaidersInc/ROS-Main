#!/usr/bin/python3

import sys
import serial
import serial.tools.list_ports
import threading
import time
from math import sin, cos, pi
import struct

import UBX
import UBXMessage

import rospy
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import Quaternion, Vector3
from nav_msgs.msg import Odometry

from ubx_utils import write_config_value, get_unique_id, get_ublox_usb_ports, read_ubx_message, cfg_valset_msg, reset_receivers, read_config_value

import UBX_consts


class gps_receiver:
    base_unique_id = b"\x2b\x49\x8a\xa6\xd8"
    rover_unique_id = b"\x2b\x49\xea\xa5\xdc"

    base_configuration = [
        [UBX_consts.CFG_USBOUTPROT_NMEA, b"\x00"],  # disable NMEA sentences
        # [UBX_consts.CFG_RATE_MEAS, b"\xC8\x01"],    # set the output rate to 5hz
        [UBX_consts.CFG_RATE_MEAS, struct.pack("<H", 1 * 1000)],    # set the output rate to 1hz
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE4072_0_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE4072_1_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1077_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1087_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1097_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1127_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1230_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE4072_0_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE4072_1_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1077_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1087_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1097_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1127_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_RTCM_3X_TYPE1230_UART2, b"\x01"],
        [UBX_consts.CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB, b"\x01"],  # absolute GPS position (latitude, longitude)
        [UBX_consts.CFG_MSGOUT_UBX_NAV_SAT_USB, b"\x01"],  # contains info about visible satellites
    ]

    rover_configuration = [
        [UBX_consts.CFG_USBOUTPROT_NMEA, b"\x00"],  # disable NMEA sentences
        # [UBX_consts.CFG_RATE_MEAS, b"\xC8\x01"],    # set the output rate to 5hz
        [UBX_consts.CFG_RATE_MEAS, struct.pack("<H", 1 * 1000)],  # set the output rate to 1hz
        [UBX_consts.CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB, b"\x01"],  # absolute GPS position (latitude, longitude)
        [UBX_consts.CFG_MSGOUT_UBX_NAV_RELPOSNED_USB, b"\x01"],  # relative position between two receivers
        [UBX_consts.CFG_MSGOUT_UBX_RXM_MEASX_USB, b"\x00"],
        [UBX_consts.CFG_MSGOUT_UBX_RXM_RTCM_USB, b"\x01"],  # output whenever the rover receiver gets a RCTM message
        [UBX_consts.CFG_MSGOUT_UBX_NAV_SAT_USB, b"\x01"],  # contains info about visible satellites
        [UBX_consts.CFG_MSGOUT_UBX_NAV_EOE_USB, b"\x01"],  # sent after each navigation "epoch"
    ]

    def __init__(self):
        self.setup_GPS()
        self.setup_ROS()
        self.rtcm_forwarder_thread = self.RTCM_forwarder(self.base_port, self.rover_port)
        self.rtcm_forwarder_thread.start()

    def setup_GPS(self):
        # reset_receivers(soft=False)
        # time.sleep(1)

        ports = get_ublox_usb_ports()
        if len(ports) != 2:
            print("wrong number of GPS recievers attached, aborting")
            sys.exit(1)

        port_1 = serial.Serial(ports[0], 9600, timeout=0.25)
        port_2 = serial.Serial(ports[1], 9600, timeout=0.25)

        # NMEA messages are really annoying and we don't need them, turn them off as soon as possible
        port_1.write(cfg_valset_msg(UBX_consts.CFG_USBOUTPROT_NMEA, b"\0x00"))
        port_2.write(cfg_valset_msg(UBX_consts.CFG_USBOUTPROT_NMEA, b"\0x00"))

        # Figure out which device is the "base" and which is the "rover". The physical units are the same, but
        # the tx2 pin of the base is connected to the rx2 pin of the rover, so if the configurations were swapped
        # it wouldn't work.
        print("about to read GPS IDs")
        ser_1_id = get_unique_id(port_1)
        ser_2_id = get_unique_id(port_2)

        if ser_1_id == self.base_unique_id and ser_2_id == self.rover_unique_id:
            self.base_port = port_1
            self.rover_port = port_2
        elif ser_1_id == self.rover_unique_id and ser_2_id == self.base_unique_id:
            self.base_port = port_2
            self.rover_port = port_1
        else:
            print("ERROR: unrecognized GPS(s) plugged in, Ublox device IDs wrong.")
            sys.exit(1)

        # time.sleep(1)
        # self.rover_port.flush()
        # self.rover_port.flushInput()
        # self.rover_port.flushOutput()
        #
        # try:
        #     print("gps_enable:", read_config_value(self.rover_port, struct.pack("<I", 0x1031001f)))
        # except:
        #     pass
        # try:
        #     print("galileo_enable:", read_config_value(self.rover_port, struct.pack("<I", 0x10310021)))
        # except:
        #     pass
        # try:
        #     print("BeiDou_enable:", read_config_value(self.rover_port, struct.pack("<I", 0x10310022)))
        # except:
        #     pass
        # try:
        #     print("QZSS_enable:", read_config_value(self.rover_port, struct.pack("<I", 0x10310024)))
        # except:
        #     pass
        # try:
        #     print("GLONASS_enable:", read_config_value(self.rover_port, struct.pack("<I", 0x10310025)))
        # except:
        #     pass

        # write the moving base configuration to the receivers
        print("writing base receiver configuration")
        for cfg_key, cfg_value in self.base_configuration:
            write_config_value(self.base_port, cfg_key, cfg_value)

        print("writing rover receiver configuration")
        for cfg_key, cfg_value in self.rover_configuration:
            write_config_value(self.rover_port, cfg_key, cfg_value)

    def setup_ROS(self):
        rospy.init_node("GPS_driver")
        self.position_publisher = rospy.Publisher("~position", NavSatFix, queue_size=2)
        self.heading_publisher = rospy.Publisher("~heading", Odometry, queue_size=2)

        # rospy.get_param('param_name')

    class RTCM_forwarder(threading.Thread):
        """
        A small class that sends all incoming data on one serial port to another serial port
        It is used to send RTCM messages from the base station GPS to the rover GPS
        """
        def __init__(self, base_port: serial.Serial, rover_port: serial.Serial):
            threading.Thread.__init__(self)
            self.base_port = base_port
            self.rover_port = rover_port

        def run(self):
            last_shutdown_check_time = time.time()
            # this code blindly forwards all data from base_port to rover_port
            # while True:
            #     waiting_chars = self.base_port.inWaiting()
            #     if waiting_chars != 0:
            #         chars = self.base_port.read(waiting_chars)
            #         self.rover_port.write(chars)
            #     else:
            #         if time.time() - last_shutdown_check_time > 0.5:
            #             if rospy.is_shutdown():  # don't call rospy.is_shutdown() more than twice a second
            #                 return
            #             last_shutdown_check_time = time.time()
            #         time.sleep(0.001)

            while True:
                while not rospy.is_shutdown():
                    binary_msgs = read_ubx_message(self.base_port)
                    msg = UBXMessage.parseUBXMessage(binary_msgs)
                    if msg._class == UBX_consts.UBX_NAV_HPPOSLLH[0] and msg._id == UBX_consts.UBX_NAV_HPPOSLLH[1]:
                        latitude = msg.lat * 10 ** (-7) + msg.latHp * 10 ** (-9)
                        longitude = msg.lon * 10 ** (-7) + msg.lonHp * 10 ** (-9)
                        height = msg.height + msg.heightHp * 10 ** (-1)
                        horiz_acc = msg.hAcc
                        vertical_acc = msg.vAcc
                        output_buffer = ("\nbase station output:\n" +
                                        "\tHPPOSLLH message received\n" +
                                          ("\t\tlatitude: %3.9f degrees  +- %3.2f m\n" % (
                                          latitude, horiz_acc * 10 ** (-3))) +
                                          ("\t\tlongitude: %3.9f degrees  +- %3.2f m\n" % (
                                          longitude, horiz_acc * 10 ** (-3))) +
                                          ("\t\theight: %3.1f meters  +- %3.2f m\n" % (
                                          height, vertical_acc * 10 ** (-3))))
                        print(output_buffer)

                    elif msg._class == UBX_consts.UBX_NAV_SAT[0] and msg._id == UBX_consts.UBX_NAV_SAT[1]:
                        # number of satellites the rover receiver can receive
                        num_aquired_signals = 0
                        # number of code and carrier locked and time synchronized satellite signals
                        num_high_quality_signals = 0
                        # number of satellites the rover receiver has differential correction data for (needed for RTK)
                        # and a high quality signal from
                        num_differential_signals = 0
                        for i in range(1, msg.numSvs + 1):
                            flags = eval("msg.flags_" + str(i))
                            qualityInd = flags % 8
                            diffCorr = flags & 2 ** 6
                            num_aquired_signals += 1 if qualityInd >= 2 else 0
                            num_high_quality_signals += 1 if qualityInd >= 5 else 0
                            num_differential_signals += 1 if diffCorr == 1 and qualityInd >= 5 else 0

                        output_buffer = "base station output:"

                        output_buffer += "\tany signal received for %3i satellites\n" % num_aquired_signals
                        output_buffer += "\tcode, carrier, and time locked/synchronized signals received from %i satelleites\n" % num_high_quality_signals
                        output_buffer += "\tdifferential correction data and high quality signal received for %i satellites\n" % num_differential_signals
                        print(output_buffer)
                    else:
                        print("base station: message received without handler: ", msg)
                    time.sleep(0.001)
                print("exiting because of base station thread")
                sys.exit(0)

    def loop(self):
        # now loop reading incoming GPS messages
        while not rospy.is_shutdown():
            recieved_RXM_RCTM = False
            RELPOSNED_message = None
            HPPOSLLH_message = None
            NAV_SAT_message = None

            while True:
                message = UBXMessage.parseUBXMessage(read_ubx_message(self.rover_port))
                if message._class == UBX_consts.UBX_NAV_HPPOSLLH[0] and message._id == UBX_consts.UBX_NAV_HPPOSLLH[1]:
                    HPPOSLLH_message = message

                elif message._class == UBX_consts.UBX_NAV_RELPOSNED[0] and message._id == UBX_consts.UBX_NAV_RELPOSNED[1]:
                    RELPOSNED_message = message

                elif message._class == UBX_consts.UBX_RXM_RTCM[0] and message._id == UBX_consts.UBX_RXM_RTCM[1]:
                    recieved_RXM_RCTM = True

                elif message._class == UBX_consts.UBX_NAV_SAT[0] and message._id == UBX_consts.UBX_NAV_SAT[1]:
                    NAV_SAT_message = message

                elif message._class == UBX_consts.UBX_NAV_EOE[0] and message._id == UBX_consts.UBX_NAV_EOE[1]:
                    break

                else:
                    print("message received without handler: ", message)

                time.sleep(0.0001)  # to let the scheduler go to the other thread

            output_buffer = "\nRover prints:\n"

            # print("Epoch summary:")
            output_buffer += "Epoch summary:\n"
            # print("\tRTCM messages received by 'rover' receiver:", recieved_RXM_RCTM)
            output_buffer += "\tRTCM messages received by 'rover' receiver: " + str(recieved_RXM_RCTM) + "\n"

            if NAV_SAT_message is not None:
                # number of satellites the rover receiver can receive
                num_aquired_signals = 0
                # number of code and carrier locked and time synchronized satellite signals
                num_high_quality_signals = 0
                # number of satellites the rover receiver has differential correction data for (needed for RTK)
                # and a high quality signal from
                num_differential_signals = 0
                for i in range(1, NAV_SAT_message.numSvs + 1):
                    flags = eval("NAV_SAT_message.flags_" + str(i))
                    qualityInd = flags % 8
                    diffCorr = flags & 2**6
                    num_aquired_signals += 1 if qualityInd >= 2 else 0
                    num_high_quality_signals += 1 if qualityInd >= 5 else 0
                    num_differential_signals += 1 if diffCorr == 1 and qualityInd >= 5 else 0

                # print("\tany signal received for %3i satellites" % num_aquired_signals)
                # print("\tcode, carrier, and time locked/synchronized signals received from %i satelleites" % num_high_quality_signals)
                # print("\tdifferential correction data and high quality signal received for %i satellites" % num_differential_signals)
                output_buffer += (("\tany signal received for %3i satellites\n" % num_aquired_signals) +
                                  ("\tcode, carrier, and time locked/synchronized signals received from %i satelleites\n" % num_high_quality_signals) +
                                  ("\tdifferential correction data and high quality signal received for %i satellites\n" % num_differential_signals))

            else:
                # print("\tNo NAV_SAT message received")
                output_buffer += "\tNo NAV_SAT message received\n"

            if HPPOSLLH_message is not None:
                latitude = HPPOSLLH_message.lat * 10**(-7) + HPPOSLLH_message.latHp * 10**(-9)
                longitude = HPPOSLLH_message.lon * 10**(-7) + HPPOSLLH_message.lonHp * 10**(-9)
                height = HPPOSLLH_message.height + HPPOSLLH_message.heightHp * 10**(-1)
                horiz_acc = HPPOSLLH_message.hAcc
                vertical_acc = HPPOSLLH_message.vAcc
                # print("\tHPPOSLLH message received")
                # print("\t\tlatitude: %3.9f degrees  +- %3.2f m" % (latitude, horiz_acc * 10**(-3)))
                # print("\t\tlongitude: %3.9f degrees  +- %3.2f m" % (longitude, horiz_acc * 10 ** (-3)))
                # print("\t\theight: %3.1f meters  +- %3.2f m" % (height, vertical_acc * 10 ** (-3)))
                output_buffer += ("\tHPPOSLLH message received\n" +
                                  ("\t\tlatitude: %3.9f degrees  +- %3.2f m\n" % (latitude, horiz_acc * 10**(-3))) +
                                  ("\t\tlongitude: %3.9f degrees  +- %3.2f m\n" % (longitude, horiz_acc * 10 ** (-3))) +
                                  ("\t\theight: %3.1f meters  +- %3.2f m\n" % (height, vertical_acc * 10 ** (-3))))

                msg = NavSatFix()
                msg.header.frame_id = "base_link"  # TODO: make a frame for the GPS
                msg.status.status = 0
                msg.status.service = 15  # defines which GNSs systems are used (eg GPS, GLONAS, GALILEO, ect)
                msg.latitude = latitude
                msg.longitude = longitude
                msg.altitude = height
                msg.position_covariance = [horiz_acc, 0, 0, 0, horiz_acc, 0, 0, 0, vertical_acc]
                msg.position_covariance_type = 2  # this means only the diagonals of the covariance matrix are known
                self.position_publisher.publish(msg)

            else:
                # print("\tNo HPPOSLLH message received, no absolute position information available")
                output_buffer += "\tNo HPPOSLLH message received, no absolute position information available\n"

            if RELPOSNED_message is not None:
                angle = RELPOSNED_message.RelPosHeading * 10 ** (-5)
                angle_accuracy = RELPOSNED_message.accHeading * 10 ** (-5)
                baseline_length = RELPOSNED_message.RelPosLength
                baseline_accuracy = RELPOSNED_message.accLength
                valid_flag = (RELPOSNED_message.flags & 2 ** 8) == 2 ** 8

                # the distance between the two receivers will never be greater than 4 meters, if the GPS(s) think
                # they're more than 1 meters appart then the fix is wrong
                baseline_good = baseline_length < 100
                # print("\tRELPOSNED message, length: %3.1f +- %3.1f (inches), angle: %-3.1f +- %2.1f, solution valid: "
                #       % (baseline_length / 2.54, baseline_accuracy / 25.4, angle,
                #          angle_accuracy) + ("yes" if valid_flag else "no"))
                output_buffer += ("\tRELPOSNED message, length: %3.1f +- %3.1f (inches), angle: %-3.1f +- %2.1f, solution valid: "
                      % (baseline_length / 2.54, baseline_accuracy / 25.4, angle,
                         angle_accuracy) + ("yes" if valid_flag else "no") + "\n")
                if not baseline_good:
                    # print("\treported baseline too large, not publishing data")
                    output_buffer += "\treported baseline too large, not publishing data\n"

                if valid_flag and baseline_good:
                    heading_msg = Odometry()
                    heading_msg.header.stamp = rospy.Time.now()
                    heading_msg.header.frame_id = "map"
                    heading_msg.child_frame_id = "base_link"
                    # -angle because the gps uses a north-east-down frame but ROS uses a east-north-up frame
                    heading_msg.pose.pose.orientation = Quaternion(0, 0, sin(-angle/180*pi/2), cos(-angle/180*pi/2))  # x, y, z, w
                    heading_msg.pose.covariance = [0, ] * 36
                    heading_msg.pose.covariance[35] = angle_accuracy/180*pi

                    heading_msg.pose.pose.position = Vector3(0, 0, 0)

                    heading_msg.twist.twist.linear = Vector3(0, 0, 0)
                    heading_msg.twist.twist.angular = Vector3(0, 0, 0)
                    heading_msg.twist.covariance = (0, ) * 36  # vector of length 9 filled with -1

                    self.heading_publisher.publish(heading_msg)

            else:
                # print("\tNo RELPOSNED message received, no relative position information available")
                output_buffer += "\tNo RELPOSNED message received, no relative position information available\n"

            print(output_buffer)
        print("exiting because of rover thread")
        sys.exit(0)


if __name__ == "__main__":
    receiver = gps_receiver()
    receiver.loop()

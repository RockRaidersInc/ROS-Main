"""
This file contains a lot of constants used in the UBX protocol (like the binary names of settings and whatnot)
"""

from struct import pack


# general stuff
START_BYTES = b"\xB5\x62"


# message class and ID
UBX_NAV_HPPOSLLH = b"\x01\x14"
UBX_NAV_RELPOSNED = b"\x01\x3C"
UBX_NAV_SAT = b"\x01\x35"
UBX_NAV_EOE = b"\x01\x61"
UBX_ACK_ACK = b"\x05\x01"
UBX_ACK_NAK = b"\x05\x00"
UBX_CFG_RST = b"\x06\x04"
UBX_CFG_VALSET = b"\x06\x8A"
UBX_CFG_VALGET = b"\x06\x8B"
UBX_RXM_RTCM = b"\x02\x32"
UBX_SEC_UNIQID = b"\x27\x03"


# configuration keys (for use with CFG-VALSET and CFG-VALGET messages)
CFG_USBOUTPROT_NMEA = pack("<I", 0x10780002)
CFG_RATE_MEAS = pack("<I", 0x30210001)
CFG_UART1_BAUDRATE = pack("<I", 0x40520001)
CFG_UART2_BAUDRATE = pack("<I", 0x40530001)
CFG_MSGOUT_RTCM_3X_TYPE4072_0_UART2 = pack("<I", 0x20910300)
CFG_MSGOUT_RTCM_3X_TYPE4072_0_USB =  pack("<I", 0x20910301)
CFG_MSGOUT_RTCM_3X_TYPE4072_1_UART2 = pack("<I", 0x20910383)
CFG_MSGOUT_RTCM_3X_TYPE4072_1_USB = pack("<I",  0x20910384)
CFG_MSGOUT_RTCM_3X_TYPE1077_UART2 = pack("<I", 0x209102ce)
CFG_MSGOUT_RTCM_3X_TYPE1077_USB = pack("<I", 0x209102cf)
CFG_MSGOUT_RTCM_3X_TYPE1087_UART2 = pack("<I", 0x209102d3)
CFG_MSGOUT_RTCM_3X_TYPE1087_USB = pack("<I", 0x209102d4)
CFG_MSGOUT_RTCM_3X_TYPE1097_UART2 = pack("<I", 0x2091031a)
CFG_MSGOUT_RTCM_3X_TYPE1097_USB = pack("<I",  0x2091031b)
CFG_MSGOUT_RTCM_3X_TYPE1127_UART2 = pack("<I", 0x209102d8)
CFG_MSGOUT_RTCM_3X_TYPE1127_USB = pack("<I", 0x209102d9)
CFG_MSGOUT_RTCM_3X_TYPE1230_UART2 = pack("<I", 0x20910305)
CFG_MSGOUT_RTCM_3X_TYPE1230_USB = pack("<I", 0x20910306)
CFG_MSGOUT_UBX_NAV_RELPOSNED_UART1 = pack("<I", 0x2091008e)
CFG_MSGOUT_UBX_NAV_RELPOSNED_USB = pack("<I", 0x20910090)
CFG_MSGOUT_UBX_NAV_SAT_USB = pack("<I", 0x20910018)
CFG_MSGOUT_UBX_RXM_MEASX_USB = pack("<I", 0x20910207)
CFG_MSGOUT_UBX_RXM_RTCM_USB = pack("<I", 0x2091026b)
CFG_MSGOUT_UBX_NAV_EOE_USB = pack("<I", 0x20910162)  # end of epoch
CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB = pack("<I", 0x20910036)

# coding=utf-8
"""Receiver Manager Messages"""

from UBXMessage import UBXMessage, initMessageClass, addGet
import struct
from Types import U1, U2, U4, X1, X2, X4, U, I1, I2, I4


@initMessageClass
class RXM:
    """Message class TIM."""

    _class = 0x02

    @addGet
    class RTCM:

        _id = 0x32

        class Fields:
            version = U1(1)
            flags = X1(2)
            subType = U2(3)
            refStation = U2(4)
            msgType = U2(5)



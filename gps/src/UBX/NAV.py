# coding=utf-8
"""Navigation result messages"""

from UBXMessage import UBXMessage, initMessageClass, addGet
import struct
from Types import U1, U2, U4, X1, X2, X4, U, I1, I2, I4


@initMessageClass
class NAV:
    """Message class NAV."""

    _class = 0x01

    @addGet
    class DOP:

        _id = 0x04

        class Fields:
            iTOW = U4(1)
            gDOP = U2(2)
            pDOP = U2(3)
            tDOP = U2(4)
            vDOP = U2(5)
            hDOP = U2(6)
            nDOP = U2(7)
            eDOP = U2(8)

    @addGet
    class HPPOSLLH:
        """
        High Precision Geodetic Position Solution
        """

        _id = 0x14

        class Fields:
            Version = U1(1)
            Reserved1_1 = U1(2)
            Reserved1_2 = U1(3)
            Flags = X1(4, allowed={1: 'invalidLlh'})
            iTOW = U4(5)
            lon = I4(6)
            lat = I4(7)
            height = I4(8)
            hMSL = I4(9)  # height above mean sea level
            lonHp = I1(10)
            latHp = I1(11)
            heightHp = I1(12)
            hMSLHp = I1(13)
            hAcc = U4(14)
            vAcc = U4(15)

    @addGet
    class SVINFO:

        _id = 0x30

        class Fields:
            iTOW = U4(1)
            numCh = U1(2)
            globalFlags = X1(3,
                                 allowed = {
                                     0 : 'Antaris',
                                     1 : 'u-Blox 5',
                                     2 : 'u-Blox 6',
                                     3 : 'u-Blox 7',
                                     4 : 'u-Blox 8',
                                     })
            reserved1 = U2(4)

            class Repeated:
                chn = U1(1)
                svid = U1(2)
                flags = X1(3)
                quality = X1(4)
                cno = U1(5)
                elev = I1(6)
                axim = I2(7)
                prRes = I4(8)
    @addGet
    class SAT:
        """
        Satellite Information
        """

        _id = 0x35

        class Fields:
            iTOW = U4(1)
            Version = U1(2)
            numSvs = U1(3)
            Reserved1 = U1(4)
            Reserved2 = U1(5)

            class Repeated:
                gnssId = U1(1)
                svId = U1(2)
                cno = U1(3)
                elev = I1(4)
                azim = I2(5)
                prRes = I2(5)
                flags = X4(6)

    @addGet
    class RELPOSNED:
        """
        Relative Positioning Information in NED frame
        """

        _id = 0x3C

        class Fields:
            Version = U1(1)
            Reserved1 = U1(2)
            RefStationId = U2(3)
            iTOW = U4(4)
            RelPosN = I4(5)
            RelPosE = I4(6)
            RelPosD = I4(7)
            RelPosLength = I4(8)
            RelPosHeading = I4(9)
            Reserved2 = U4(10)
            RelPosHPN = I1(11)
            RelPosHPE = I1(12)
            RelPosHPD = I1(13)
            relPosHPLength = I1(14)
            accN = U4(15)
            accE = U4(16)
            accD = U4(17)
            accLength = U4(18)
            accHeading = U4(19)
            Reserved3 = U4(20)
            flags = X4(21,
                            allowed = {
                                     0 : 'gnssFixOK',
                                     1 : 'diffSoln',
                                     2 : 'relPosValid',
                                     3 : 'carrSoln1',
                                     4 : 'carrSoln2',
                                     5 : 'isMoving',
                                     6 : 'refPosMiss',
                                     7 : 'refObsMiss',
                                     8 : 'relPosHeadingValid',
                                     9 : 'relPosNormalized'
                                     })

    @addGet
    class EOE:
        """
        End Of Epoch
        """

        _id = 0x61

        class Fields:
            iTow = U4(1)

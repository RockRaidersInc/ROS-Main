"""Security result messages"""

from UBXMessage import UBXMessage, initMessageClass, addGet
from Types import U1


@initMessageClass
class SEC:
    """Message class SEC."""

    _class = 0x27

    @addGet
    class UNIQID:

        _id = 0x03

        class Fields:
            version = U1(1)
            reserved1_1 = U1(2)
            reserved1_2 = U1(3)
            reserved1_3 = U1(4)
            uniqueId_1 = U1(5)
            uniqueId_2 = U1(6)
            uniqueId_3 = U1(7)
            uniqueId_4 = U1(8)
            uniqueId_5 = U1(9)

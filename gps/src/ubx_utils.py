import struct
import serial

import UBXMessage

import timeout_decorator

import UBX_consts


class NakException(Exception):
    pass


def write_config_value(device: serial.Serial, key: bytes, value: bytes):
    set_msg = format_message(UBX_consts.UBX_CFG_VALSET,
                             [b"\x00",  # message version, set to 0 for a single key/value pair
                              b"\x07",  # where to save the configuration. 7 for all layers
                              b"\x00\x00",  # reserved
                              key, value])
    device.write(set_msg)

    # read the configuration value back to make sure it was written correctly
    get_msg = format_message(UBX_consts.UBX_CFG_VALGET,
                             [b"\x00",  # message version, set to 0 for a single key/value pair
                              b"\x01",  # what layer to read configuration from
                              b"\x00\x00",  # reserved
                              key])

    get_response = write_command_get_response_retry(device, get_msg)
    # Using pyUBX to parse CFG-GET messages would be really hard because config values can be variable length, depending
    # on the key (and I have no idea how to make the length of a message component depend on other parts of the message.
    # The config key and value are always at the same location in the response message, so we can just grab them out.
    responded_key = get_response[10:14]
    responded_val = get_response[14:14 + len(value)]
    if responded_key != key:
        raise Exception("wrong configuration key sent back as response, a serial port probably needs to be flushed")
    if responded_val != value:
        raise Exception("writing configuration to GPS failed")


def read_config_value(device: serial.Serial, key: bytes):
    # read the configuration value back to make sure it was written correctly
    get_msg = format_message(UBX_consts.UBX_CFG_VALGET,
                             [b"\x00",  # message version, set to 0 for a single key/value pair
                              b"\x01",  # what layer to read configuration from
                              b"\x00\x00",  # reserved
                              key])

    print(UBXMessage.formatByteString(get_msg))

    get_response = write_command_get_response_retry(device, get_msg)
    # Using pyUBX to parse CFG-GET messages would be really hard because config values can be variable length, depending
    # on the key (and I have no idea how to make the length of a message component depend on other parts of the message.
    # The config key and value are always at the same location in the response message, so we can just grab them out.
    responded_key = get_response[10:14]
    responded_val = get_response[14:len(get_response) - 2]
    print(UBXMessage.formatByteString(get_response))
    if responded_key != key:
        raise Exception("wrong configuration key sent back as response, a serial port probably needs to be flushed")
    return responded_val


def reset_receivers(soft=False):
    """
    Sends a UBX-CFG-RST message to the receiver. A cold navigation reset is always requested.
    :param soft: if soft==True then a "controlled software reset" is done. If soft==False then hardware reset is done.
    :return: None
    """
    if soft:
        reset_command = UBX_consts.UBX_CFG_RST + b"\x04\x00\xFF\xFF\x01\x00"
    else:
        reset_command = UBX_consts.UBX_CFG_RST + b"\x04\x00\xFF\xFF\x00\x00"

    ports = get_ublox_usb_ports()
    if len(ports) != 2:
        print("wrong number of GPS recievers attached, aborting")
        return

    ser_1 = serial.Serial(ports[0], 9600, timeout=None)
    ser_2 = serial.Serial(ports[1], 9600, timeout=None)

    # send reset to clear whatever state the GPS recievers are in
    ser_1.write(add_start_bits_checksum(reset_command))
    ser_2.write(add_start_bits_checksum(reset_command))

    # now re-open the serial ports:
    ser_1.close()
    ser_2.close()


def get_unique_id(device):
    get_unique_id_command = format_message(UBX_consts.UBX_SEC_UNIQID, [])
    response = UBXMessage.parseUBXMessage(write_command_get_response(device, get_unique_id_command))
    device_id = bytes([response.uniqueId_1,
                        response.uniqueId_2,
                        response.uniqueId_3,
                        response.uniqueId_4,
                        response.uniqueId_5])
    return device_id

def get_ublox_usb_ports():
    """
    Returns a list of serial ports connected to Ublox gps receivers
    :return:
    """
    return [i.device for i in serial.tools.list_ports.comports() if i.product == "u-blox GNSS receiver"]


def cfg_valset_msg(key: bytes, val, layer: int=1) -> bytes:
    if type(val) is not bytes:
        val_bytes = struct.pack("<B", val)
    else:
        val_bytes = val
    layer_bytes = struct.pack("<B", layer)
    # return Message(b"\x06", b"\x8A", [b"\x00", layer_bytes, b"\x00\x00", key, val_bytes])
    return format_message(UBX_consts.UBX_CFG_VALSET, [b"\x00", layer_bytes, b"\x00\x00", key, val_bytes])


def format_message(class_id_bytes: bytes, payload: list):
    """
    Puts message parts into a single byte array with the start bits and checksum.
    :param class_bytes: Message class (for example, 0x06 for UBX-CFG)
    :param id_bytes: Message ID/type (for example, 0x01 for UBX-ACK-ACK
    :param payload: Message payload. The message length should not be included.
    :return:
    """
    concatenated_payload = b""
    for i in payload:
        concatenated_payload += i
    message_body = class_id_bytes + struct.pack("<H", len(concatenated_payload)) + concatenated_payload
    return add_start_bits_checksum(message_body)


def calc_checksum(full_message):
    CK_A = 0
    CK_B = 0
    for byte in full_message:
        CK_A = 0xff & (CK_A + byte)
        CK_B = 0xff & (CK_B + CK_A)
    return CK_A, CK_B


def add_start_bits_checksum(msg):
    A, B = calc_checksum(msg)
    return b"\xB5\x62" + msg + bytes([A]) + bytes([B])


def write_command_get_response_retry(device: serial.Serial, command: bytes, tries=4) -> bytes:
    """
    Same as write_command_get_response() but has a timeout and it will re-send the sent message up to
    tries times if the correct response is not received.
    :param device:
    :param command:
    :param tries:
    :return:
    """
    for i in range(tries):
        try:
            return __write_command_get_response_with_timeout(device, command)
        except StopIteration as e:
            continue
    raise Exception("command failed 4 times in a row")


@timeout_decorator.timeout(0.5, timeout_exception=StopIteration)
def __write_command_get_response_with_timeout(device: serial.Serial, command: bytes) -> bytes:
    return write_command_get_response(device, command)


def write_command_get_response(device: serial.Serial, command: bytes) -> bytes:
    """
    Writes the given command to the device then returns the response. Responses must be of the same message type.
    :param device: a serial port
    :param command:
    :return:
    """
    device.flush()
    device.flushInput()
    device.flushOutput()
    device.write(command)

    while True:
        device.flush()
        msg = read_ubx_message(device)
        # see if the received message class and type match the passed command
        if msg[0:4] == UBX_consts.START_BYTES + UBX_consts.UBX_ACK_NAK:
            raise NakException("NAK received after sending command " + UBXMessage.formatByteString(msg))
        if msg[0:4] == command[0:4]:
            return msg


def read_ubx_message(device: serial.Serial) -> bytes:
    """
    Read a single UBX message from the passed serial port and return it.
    Any partial messages or non-UBX messages will be ignored
    :param device:
    :return:
    """
    while True:
        # every UBX message starts with the bytes b"\xB5\x62", look for them in the serial input
        start_byte_1 = device.read(1)
        if start_byte_1 == UBX_consts.START_BYTES[0:1]:
            start_byte_2 = device.read(1)
            if start_byte_2 == UBX_consts.START_BYTES[1:2]:
                # we are now in a message, read the rest of it
                class_type = device.read(2)
                length_packed = device.read(2)
                try:
                    length = struct.unpack("<H", length_packed)[0]
                except struct.error as e:
                    continue  # this probably means that not enough characters were read. Just skip to the next message
                msg_body = device.read(length)
                checksum = device.read(2)

                # make sure the checksum is valid
                calculated_msg = format_message(class_type[0:2], [msg_body])
                if calculated_msg != UBX_consts.START_BYTES + class_type + length_packed + msg_body + checksum:
                    # raise Exception("msg checksum did not match")
                    print("GPS receiver: got corrupted message")
                    continue

                # checksum did match, return the message
                return calculated_msg
        # else:
        #     print(chr(start_byte_1[0]), end="")

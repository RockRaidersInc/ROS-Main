import socket

class BaseStation:
    RECV_BUFF = 1024

    def __init__(remote_ipaddr, remote_port, local_ipaddr, local_port):
        this.ipaddr = remote_ipaddr
        this.port = remote_port

        this.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        this.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        this.sock_recv.bind((local_ipaddr, local_port))

        while not rospy.is_shutdown():
            rospy.sleep(1)

    def send_message(msg):
        this.sock_send.sendto(msg, (ipaddr, port))

    def receive_message():
        data, addr = this.sock_recv.recvfrom(RECV_BUFF)
        return data
        
        
        

if __name__ == '__main__':
    rospy.init_node('baseStation', anonymous=True)
    baseStation = BaseStation('192.168.1.99', 5000, '192.168.1.1', 5000)

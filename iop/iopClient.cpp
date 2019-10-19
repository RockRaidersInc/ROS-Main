#include <iostream>  
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 3794
#define MAX_BUFFER 256


int findCVT() 
{
	int				sockfd, n;
	struct sockaddr_in	servaddr;

	bzero(&servaddr, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT);
	inet_pton(AF_INET, "239.255.0.1", &servaddr.sin_addr);

	sockfd = socket(AF_INET, SOCK_DGRAM, 0);

	char buffer[MAX_BUFFER];
	while (n < 0) {
		n = recvfrom( sockfd, &buffer, MAX_BUFFER, 0, (sockaddr *)&servaddr, (socklen_t*) sizeof(&servaddr));
	}
	printf("recieved %d bytes\n", n);
	printf("%s\n", buffer);



}



int main(int argc, char **argv)
{
	int sd = findCVT();
	printf("sd is %d\n", sd);

	exit(0);
}

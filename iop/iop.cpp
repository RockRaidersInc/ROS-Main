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

#define PORT 9877
#define MAX_BUFFER 256

enum {RESERVED, SYSTEM, SUBSYSTEM, NODE, COMPONENT};
/* 
* JTC will broadcast QueryIdentification message using 239.255.0.1 Multicast, every 5 seconds until response
* Checks that the response has an appropriate subsystem name
* Then sends direct query, expecting the same response
* 
* 
*/ 
int QueryIdentification() 
{
	int	sd;
	struct sockaddr_in server, client, bcast;
	struct in_addr localInterface;

	if ( (sd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 

    // memset(&server, 0, sizeof(server)); 
    // memset(&client, 0, sizeof(client)); 

    // server.sin_family    = AF_INET; // IPv4 
    // server.sin_addr.s_addr = INADDR_ANY; 
    // server.sin_port = htons(PORT);

    // bind(sd, (sockaddr *)&server, sizeof(server));

    // Broadcast address
    bcast.sin_family    = AF_INET; // IPv4 
    bcast.sin_addr.s_addr = inet_addr("239.255.0.1"); 
    bcast.sin_port = htons(PORT)


    /*
    * Disable loopback so you do not receive your own datagrams.
    */
    char loopch=0;

    if (setsockopt(sd, IPPROTO_IP, IP_MULTICAST_LOOP,
                   (char *)&loopch, sizeof(loopch)) < 0) {
      perror("setting IP_MULTICAST_LOOP:");
      close(sd);
      exit(1);
    }
    

    // Client address, will be populated when we receive data
    client.sin_family = AF_INET;
    client.sin_addr.s_addr = INADDR_ANY;
    char opts = 0;
	setsockopt(sd, IPPROTO_IP, IP_MULTICAST_LOOP, (char*)&opts, sizeof(opts));
	localInterface.s_addr = inet_addr("9.5.1.1");
	if (setsockopt(sd, IPPROTO_IP, IP_MULTICAST_IF,
                 (char *)&localInterface,
                 sizeof(localInterface)) < 0) 
    {
    perror("setting local interface");
    exit(1);
    }

    datalen = 10;
    if (sendto(sd, databuf, datalen, 0,
                 (struct sockaddr*)&groupSock,
                 sizeof(groupSock)) < 0)
    {
        perror("sending datagram message");
    }


    // Look for the message from the thing
    int len, n = -1; 
    char buffer[MAX_BUFFER];
    char msg[] = "QueryIdentification";
    while (n < 0) 
    {
    	sendto(sd, msg, sizeof(msg), 0, (struct sockaddr*) &bcast, (socklen_t)sizeof(&bcast));

    	n = recvfrom(sd, &buffer, MAX_BUFFER, 0, (sockaddr*)&client, (socklen_t *)sizeof(&client));
    	printf("n is %d\n", n);

    	sleep(5);
	}

	printf("IP address is: %s\n", inet_ntoa(client.sin_addr));
	return(sd);
}

int main(int argc, char *argv[]) 
{

	printf("START?\n");
	int sd = QueryIdentification();
	printf("Got %d for an sd\n", sd);
	
}
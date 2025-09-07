CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

all: main

main: main.c
	$(CC) $(CFLAGS) main.c -o main $(LDFLAGS)

clean:
	rm -f main


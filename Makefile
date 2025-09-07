# Try to use gcc, fallback to clang if not found
ifeq (, $(shell command -v gcc 2>/dev/null))
    CC := clang
else
    CC := gcc
endif

CFLAGS = -O3 -Wall
LDFLAGS = -lm

all: main

main: main.c
	$(CC) $(CFLAGS) main.c -o main $(LDFLAGS)

clean:
	rm -f main


# lib/Makefile
CC     = gcc
CFLAGS = -O2 -Wall -fPIC
TARGET = linear_model.so

all: $(TARGET)

$(TARGET): linear_model.c linear_model.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) linear_model.c

clean:
	rm -f $(TARGET)
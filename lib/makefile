CC = gcc
CFLAGS = -Wall -O2 -fPIC
LDFLAGS = -shared

all: linear_model.so mlp.so

linear_model.so: linear_model.o
	$(CC) $(LDFLAGS) -o $@ $^

mlp.so: mlp.o
	$(CC) $(LDFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o *.so

install:
	cp *.so ../python/

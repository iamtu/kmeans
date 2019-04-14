
CC = g++

CFLAGS = -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall
LDFLAGS = -lpthread

OBJS = $(addprefix build/, main.o common.o kmeans_io.o kmeans.o)

semtp: $(OBJS)
	$(CC) $(CFLAGS) -o kmeans $(OBJS) $(LDFLAGS)

build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MM -MT build/$*.o $< > build/$*.d
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	-rm kmeans
	-rm -rf build

-include build/*.d

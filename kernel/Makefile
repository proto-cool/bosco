# Deterministic build flags: no fast-math, no FMA contraction, strict FP.
CC ?= cc
CFLAGS ?= -O3 -std=c99 -fPIC -Wall -Wextra -fno-fast-math -ffp-contract=off -fexcess-precision=standard -fno-trapping-math
liblif.so: lif.c lif.h
	$(CC) $(CFLAGS) -shared -o $@ lif.c -lm
clean:
	rm -f liblif.so
.PHONY: clean

.PHONY: all build sqush clean format

all: build

build:
	./build.sh

sqush: clean
	./enroot.sh -n cuda -f ${PWD}/Dockerfile

clean:
	rm -rf build/

format:
	find . -type f -name "*.cu" -o -name "*.h" -o -name "*.cuh"| xargs -I{} clang-format -style=file -i {}
	black scripts/*

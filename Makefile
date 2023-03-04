.PHONY: all test format docs clean
all: format test docs clean

test:
	pytest --cov cri --cov-append

format:
	isort cri examples tests
	black cri examples tests
	ruff check cri examples tests
	mypy cri examples tests

docs:
	cd docs
	make clean doctest html
	cd ..

clean:
	coverage erase

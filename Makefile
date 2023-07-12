.PHONY: all test format docs clean
all: format test docs clean

test:
	pytest --cov confounding_robust_inference --cov-append --cov-report html

format:
	isort confounding_robust_inference examples tests
	black confounding_robust_inference examples tests
	ruff check confounding_robust_inference examples tests
	mypy confounding_robust_inference examples tests

docs:
	cd docs;make clean doctest html

clean:
	coverage erase

isort:
	isort .

black:
	black .

flake8:
	flake8 .

format: isort black flake8

test:
	python -m coverage run -m pytest tests/ && \
	python -m coverage report -m
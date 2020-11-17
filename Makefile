ROOT_PATH=./sparsemod/

clean-pycache:
	find . -type f -name "__pycache__" -exec rm -rf {} \;
	
clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

style:
	black --line-length 79 --target-version py37 examples torch2cmsis
	isort --recursive examples torch2cmsis

quality:
	-pylint torch2cmsis examples


PHONY: style quality  clean-pycache
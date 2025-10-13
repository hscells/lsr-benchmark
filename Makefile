build-pypi-package: run-tests
	rm -Rf dist
	python3 -m build --sdist .
	python3 -m build --wheel .
        twine upload dist/lsr_benchmark-*-py3-none-any.whl dist/lsr_benchmark-*.tar.gz

run-tests:
	pytest test

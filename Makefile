check_dirs := .

quality:
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
feat:
	PYTHONPATH=./src python bin/featuring_knn_model.py

reg-knn:
	PYTHONPATH=./src python bin/detect_using_trained_model.py

reg:
	PYTHONPATH=./src python bin/detect_using_input_image.py

# Cleaning up the python compiled bytecodes
clear-pyc:
    find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

setup:
	pip install -r requirements.txt
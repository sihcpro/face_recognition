feat:
	PYTHONPATH=./src python bin/featuring_knn_model.py

reg-knn:
	PYTHONPATH=./src python bin/detect_using_trained_model.py

reg:
	PYTHONPATH=./src python bin/detect_using_input_image.py

eval:
	PYTHONPATH=./src python bin/evaluate.py

setup:
	pip install -r requirements.txt
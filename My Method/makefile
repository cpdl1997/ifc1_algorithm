all: 
dataset_create
run_model_sequential
run_model_mlp

dataset_create: src\dataset_gen\dataset_generator.py Makefile
	cd src/dataset_gen
	python3 dataset_generator.py

run_model_sequential: src\model\custom_sequential.py Makefile
	cd src/model
	python3 custom_sequential.py

run_model_mlp: src\model\mlp_classifier.py Makefile
	cd src/model
	python3 mlp_classifier.py

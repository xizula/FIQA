pypath:
	export PYTHONPATH=.

get_embeddings:
	python src/face_reco/face_reco.py

memory:
	sudo echo 1 > /proc/sys/vm/overcommit_memory
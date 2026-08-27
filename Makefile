.PHONY: train demo dashboard test compose

train:            ## train + register the MLlib anomaly model (writes data/model)
	python -m stream.train_model

demo:             ## end-to-end local run (file source): stream -> alerts -> incidents
	python scripts/run_local_demo.py

dashboard:        ## serve the ops dashboard locally
	streamlit run dashboard/app.py

test:
	pytest tests/ -q

compose:          ## bring up the full Kafka + Spark + dashboard stack
	docker compose -f docker/docker-compose.yml up --build

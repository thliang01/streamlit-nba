.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py
run-container:
	@docker build -f Dockerfile -t app:latest .
	@docker run -p 8501:8501 app:latest

gcloud-deploy:
	@gcloud app deploy app.yaml
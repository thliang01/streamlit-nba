.PHONY: run run-container gcloud-deploy

run:
	@streamlit run app.py
run-container:
	@docker build -f Dockerfile -t streamlit-nba .
	@docker run -p 8080:8080 -e PORT=8080 streamlit-nba


gcloud-deploy:
	@gcloud app deploy app.yaml
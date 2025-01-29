.PHONY: run_cron
.PHONY: serve

run_cron:
	python3 cron/main.py

serve:
	python3 web/flask_app.py
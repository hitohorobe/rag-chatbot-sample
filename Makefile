init:
	@poetry install
	@poetry shell

setup-db:
	@python setup/setup_from_pdf.py

run:
	@streamlit run app/main.py
# Reproduce the grocery-oligopoly project end to end.
#   make setup     create .venv and install Python + Node dependencies
#   make data      download Project Hammer + StatCan raw files (~4 GB unzipped)
#   make all       clean -> analyse -> figures -> workbook -> deck -> memo
PY      ?= .venv/bin/python
SRC      = src

.PHONY: setup data panel analysis figures workbook deck memo all clean

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
	cd deck && npm install --silent

data:
	$(PY) $(SRC)/download.py

panel:
	$(PY) $(SRC)/build_panel.py

analysis:
	$(PY) -W ignore $(SRC)/analysis.py

figures:
	$(PY) $(SRC)/figures.py

workbook:
	$(PY) $(SRC)/workbook.py

deck:
	cd deck && node build_deck.js

memo:
	pandoc deliverables/research_memo.md --resource-path=deliverables -o deliverables/research_memo.docx

all: panel analysis figures workbook deck memo

clean:
	rm -rf data/processed/* outputs/tables/*.csv outputs/figures/*.png

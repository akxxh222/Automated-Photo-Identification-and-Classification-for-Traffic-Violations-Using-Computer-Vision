.PHONY: setup train demo dashboard eval clean

TRAIN_ARGS =
ifeq ($(SKIP),1)
	TRAIN_ARGS = --skip-train
endif

setup:
	pip install -r requirements.txt
	mkdir -p datasets models results evidence_store reports tests logs

train:
	python scripts/train_all.py $(TRAIN_ARGS)

demo:
	python scripts/run_demo.py

dashboard:
	streamlit run app/app.py

eval:
	python scripts/evaluate_all.py

api:
	uvicorn src.api.app:app --reload --port 8000

clean:
	@echo "No cleanup action configured."

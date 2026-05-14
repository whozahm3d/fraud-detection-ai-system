.PHONY: install install-dev run notebook-final notebook-rag notebook-eda lint clean help

# ── Configuration ────────────────────────────────────────────────────────────
PYTHON      := python3
PIP         := pip
STREAMLIT   := streamlit
JUPYTER     := jupyter notebook
APP         := app.py

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  TrustGuard AI — Available Commands"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make install        Install all runtime dependencies"
	@echo "  make install-dev    Install runtime + dev dependencies"
	@echo "  make run            Launch the Streamlit dashboard"
	@echo "  make notebook-final Open the final pipeline notebook (Deliverable 3)"
	@echo "  make notebook-rag   Open the RAG system notebook"
	@echo "  make notebook-eda   Open the EDA notebook (Deliverable 1)"
	@echo "  make lint           Run flake8 linter on source files"
	@echo "  make clean          Remove Python cache files"
	@echo "  ─────────────────────────────────────────────────────"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

# ── Run ───────────────────────────────────────────────────────────────────────
run:
	$(STREAMLIT) run $(APP)

# ── Notebooks ─────────────────────────────────────────────────────────────────
notebook-final:
	$(JUPYTER) notebooks/final_deliverable_notebooks/trustguard_ai_final.ipynb

notebook-rag:
	$(JUPYTER) notebooks/final_deliverable_notebooks/rag_system.ipynb

notebook-eda:
	$(JUPYTER) notebooks/deliverable_1_notebooks/exploratory_data_analysis.ipynb

# ── Code Quality ──────────────────────────────────────────────────────────────
lint:
	flake8 $(APP) rag_module.py scripts/ \
		--max-line-length=120 \
		--exclude=__pycache__ \
		--statistics

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "✅ Cache files removed."

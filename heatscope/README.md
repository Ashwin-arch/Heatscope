# HeatScope Demo

This is the **single Streamlit demo file** for the existing HeatScope project.

Save:
`heatscope/demo.py`

Run:
```bash
cd /home/ashwin/conference/HeatScope
source .venv/bin/activate
pip install streamlit plotly
python -m streamlit run heatscope/demo.py
```

The app connects to:
- `results/pareto_frontier.csv`
- `data/` / `results/` master spatial files when present
- the existing HeatScope project root

The model benchmark, SHAP, uncertainty, optimization and Pareto values shown in the interface
are the validated HeatScope study results already produced during the project.

Recommended presentation flow:
Executive View -> Heat Intelligence -> Model Performance -> Explainability ->
Uncertainty -> Intervention Planner (75% / 80%) -> Pareto -> Viva Defense.

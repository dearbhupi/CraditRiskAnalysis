pip install uv
uv init
uv venv --python 3.13.7  
source .venv/bin/activate
uv add pandas
uv init
uv add pandas
uv add streamlit
uv add scikit-lean matplotlib jupyterlab
uv add pandas numpy scikit-lean matplotlib jupyterlab
uv add pandas numpy scikit-learn matplotlib jupyterlab
git init
git branch -M main
curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore
git add .
git commit -m "inital commit "
git remote add origin https://github.com/dearbhupi/CraditRiskAnalysis.git
git push -u origin main
git status
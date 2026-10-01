"""Entry point for hosting the research assistant on Streamlit Community Cloud.

Community Cloud installs the packages listed in requirements.txt beside this file (the app's
own short list; the repository's root requirements.txt is the larger notebook environment),
then runs this file. It simply runs the real app, scripts/assistant_app.py, so there is one
app to maintain. Locally, `streamlit run scripts/assistant_app.py` still works as before.
"""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "assistant_app.py"), run_name="__main__")

# CVNL-Assignment
## Set Up
### In the CVNL-Assignment folder:
1. Create a virtual environment (Optional but recommended, if failed can be skipped)
A virtual environment isolates project dependencies to avoid conflicts with other Python installations. It is optional because the project can run using globally installed packages

For Windows:
python -m venv .venv
.venv\Scripts\activate.bat
For macOS:
python3 -m venv .venv
source .venv/bin/activate

2. Install required packages
pip install streamlit torch torchvision pillow numpy scikit-learn

3. Run the application
streamlit run app.py

4. To stop the application
Ctrl + C
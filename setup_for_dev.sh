# If no commits exist, initialise the repo
if ! git rev-parse HEAD > /dev/null 2>&1; then
  git init
  git add .
  git commit -m "init: first commit"
fi

# Check if Python 3.9 is installed
if command -v python3.9 &> /dev/null; then
    # Python 3.9 is installed, use it to run the script
    python3.9 -m venv .venv
else
    # Python 3.11 is not installed, use the default Python version
    python -m venv .venv
fi

source .venv/bin/activate

pip install -e .[dev,tests]

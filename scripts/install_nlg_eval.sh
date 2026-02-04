#!/usr/bin/env bash
# Install patched gensim 3.8.3 (fixes __NUMPY_SETUP__ build error on Python 3.8+),
# then install nlg-eval. Run with venv38 activated:
#   source venv38/bin/activate
#   bash scripts/install_nlg_eval.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GENSIM_DIR="${TMPDIR:-/tmp}/gensim-3.8.3-$$"

echo "Cloning gensim 3.8.3..."
git clone --depth 1 --branch 3.8.3 https://github.com/RaRe-Technologies/gensim.git "$GENSIM_DIR"
cd "$GENSIM_DIR"

echo "Patching setup.py for Python 3.8+ / pip build isolation..."
python3 << 'PATCH'
with open("setup.py", "r") as f:
    lines = f.readlines()
patched = []
i = 0
while i < len(lines):
    line = lines[i]
    # Find the line (may have leading whitespace)
    if "__builtins__.__NUMPY_SETUP__ = False" in line and "try:" not in line:
        indent = line[: len(line) - len(line.lstrip())]
        block = (
            indent + "try:\n"
            + indent + "    __builtins__.__NUMPY_SETUP__ = False\n"
            + indent + "except AttributeError:\n"
            + indent + "    pass\n"
        )
        patched.append(block)
        i += 1
    else:
        patched.append(line)
        i += 1
with open("setup.py", "w") as f:
    f.writelines(patched)
print("Patched setup.py")
PATCH

echo "Installing patched gensim 3.8.3..."
pip install .

cd "$PROJECT_ROOT"
rm -rf "$GENSIM_DIR"

echo "Installing nlg-eval..."
pip install git+https://github.com/Maluuba/nlg-eval.git@master

echo "Downloading spacy model en_core_web_sm..."
python -m spacy download en_core_web_sm

echo "Done. nlg-eval is installed."

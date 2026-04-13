"""
Launcher for AI Risk — Streamlit app.

Compiled to AI Risk.exe via build_exe.bat.
Searches for Python in the project virtual environment first, then the system
PATH. Verifies Streamlit is available before handing off.
"""
import os
import shutil
import subprocess
import sys


def _find_python(here: str) -> tuple[str | None, list[str]]:
    """Return (resolved_python_path, searched_paths_list).

    Search order:
      1. <project>/.venv/Scripts/python.exe  (preferred — project venv)
      2. <project>/venv/Scripts/python.exe   (alternate venv name)
      3. python on the system PATH
      4. python3 on the system PATH
    """
    candidates = [
        os.path.join(here, ".venv", "Scripts", "python.exe"),
        os.path.join(here, "venv",  "Scripts", "python.exe"),
        shutil.which("python"),
        shutil.which("python3"),
    ]

    searched: list[str] = []
    for path in candidates:
        if not path:
            continue
        label = path if os.path.isabs(path) else os.path.abspath(path)
        searched.append(label)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path, searched

    return None, searched


def _streamlit_available(python: str) -> bool:
    """Return True if `python -m streamlit --version` exits cleanly."""
    try:
        result = subprocess.run(
            [python, "-m", "streamlit", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except Exception:
        return False


def _fail(message: str) -> None:
    print(message)
    input("\nPressione Enter para sair / Press Enter to exit...")
    sys.exit(1)


def main() -> None:
    # ── 1. Locate the project root (same folder as the exe or this script). ──
    if getattr(sys, "frozen", False):
        here = os.path.dirname(os.path.abspath(sys.executable))
    else:
        here = os.path.dirname(os.path.abspath(__file__))

    os.chdir(here)

    # ── 2. Verify app.py is present. ──
    app = os.path.join(here, "app.py")
    if not os.path.exists(app):
        _fail(
            f"Erro: app.py não encontrado em '{here}'.\n"
            f"Error: app.py not found in '{here}'.\n"
            "O executável deve ficar na mesma pasta do projeto.\n"
            "The executable must be placed in the project folder."
        )

    # ── 3. Resolve Python interpreter. ──
    python, searched = _find_python(here)
    if python is None:
        searched_list = "\n  ".join(searched) if searched else "  (nenhum caminho verificado)"
        _fail(
            "Erro: nenhum interpretador Python utilizável foi encontrado.\n"
            "Error: no usable Python interpreter was found.\n"
            "\nLocais verificados / Locations searched:\n  " + searched_list + "\n"
            "\nSoluções / Solutions:\n"
            "  • Crie o ambiente virtual:  python -m venv .venv\n"
            "  • Ou instale Python:        https://www.python.org/downloads/\n"
            "  • Create the virtual env:  python -m venv .venv\n"
            "  • Or install Python:       https://www.python.org/downloads/"
        )

    # ── 4. Verify Streamlit is installed in the resolved environment. ──
    if not _streamlit_available(python):
        _fail(
            f"Erro: Python encontrado em '{python}'\n"
            f"Error: Python found at '{python}'\n"
            "mas o Streamlit não está disponível nesse ambiente.\n"
            "but Streamlit is not available in that environment.\n"
            "\nInstale as dependências / Install dependencies:\n"
            f"  {python} -m pip install -r requirements.txt"
        )

    # ── 5. Launch the app. ──
    result = subprocess.run([python, "-m", "streamlit", "run", app])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

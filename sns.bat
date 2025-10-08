@echo off
call make-venv-uv.bat > NUL 2>&1
python sns.py %*
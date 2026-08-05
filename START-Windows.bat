@echo off
rem ---------------------------------------------------------------------
rem This file must stay pure ASCII. cmd.exe reads batch files in the
rem system codepage, so non-ASCII text is mangled, and a stray byte that
rem looks like ")" ends an if-block early and the rest of the message gets
rem executed as commands. Labels are used instead of parenthesised blocks
rem for the same reason.
rem ---------------------------------------------------------------------
cd /d "%~dp0"
title TEM Particle Analyzer

echo ============================================
echo    TEM Particle Analyzer
echo ============================================
echo.

if not exist "run.py" goto NOFILES

rem "python" alone can resolve to the Microsoft Store stub, which exits
rem without running anything, so each candidate must prove it runs code.
set "PY="
py -3 -c "print(1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python -c "print(1)" >nul 2>&1 && set "PY=python"
if not defined PY python3 -c "print(1)" >nul 2>&1 && set "PY=python3"
if defined PY goto HAVEPYTHON

rem The Python Install Manager provides "py" but ships no runtime until
rem "py install" is run, so "py" exists yet cannot execute any code.
where py >nul 2>&1 && goto NORUNTIME
goto NOPYTHON

:HAVEPYTHON

echo Python found:
%PY% --version
echo.

echo Installing required packages. The first run may take a few minutes...
%PY% -m pip install --user --quiet --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto PIPFAIL

echo.
echo Starting the program. Keep this black window open.
echo.
%PY% run.py
if errorlevel 1 goto RUNFAIL
exit /b 0


:NOFILES
echo [ERROR] run.py not found in this folder.
echo.
echo This file must sit in the same folder as run.py and requirements.txt.
echo If you unzipped the download, open the inner folder and run it there.
echo.
pause
exit /b 1

:NORUNTIME
echo [ERROR] Python Install Manager is present, but no Python runtime
echo         is installed yet.
echo.
echo Run this command, wait for the download to finish, then start
echo this file again:
echo.
echo     py install 3.13
echo.
echo You can check it worked with:  py list
echo.
pause
exit /b 1

:NOPYTHON
echo [ERROR] Python is not installed.
echo.
echo   1. Go to https://www.python.org/downloads/
echo   2. Click the yellow "Download Python" button
echo   3. Run the installer
echo   4. IMPORTANT: tick "Add python.exe to PATH" on the first screen
echo   5. Click "Install Now", then run this file again
echo.
pause
exit /b 1

:PIPFAIL
echo.
echo [ERROR] Package installation failed.
echo Check your internet connection. On a corporate network a firewall or
echo proxy may be blocking it - ask IT for the internal PyPI mirror address.
echo.
pause
exit /b 1

:RUNFAIL
echo.
echo [ERROR] The program failed to start.
echo Copy the messages above and send them for help.
echo.
pause
exit /b 1

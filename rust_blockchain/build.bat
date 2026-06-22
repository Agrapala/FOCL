@echo off
echo =================================================
echo  FLoBC Rust Blockchain Node - Build Script
echo  Requires: rustup + cargo (https://rustup.rs)
echo =================================================
echo.
where rustc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: Rust not found. Install from https://rustup.rs
    echo  After install, restart terminal and run this script again.
    pause
    exit /b 1
)
echo  Rust found:
rustc --version
cargo --version
echo.
echo  Building release binary...
cargo build --release
if %ERRORLEVEL% EQU 0 (
    echo.
    echo  Build SUCCESS!
    echo  Binary: target\release\flobc-blockchain.exe
    echo.
    echo  To run the blockchain node:
    echo    target\release\flobc-blockchain.exe
    echo.
    echo  Or let run_network.py start it automatically.
) else (
    echo.
    echo  Build FAILED. Check errors above.
)
pause

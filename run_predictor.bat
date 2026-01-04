@echo off
chcp 65001
echo ==========================================
echo 독립형 장애 예측기를 시작합니다...
echo ==========================================

:: 브라우저 자동 실행 (2초 대기 후 실행)
timeout /t 5 >nul
start http://localhost:8001

:: 프로그램 실행
python main.py

pause

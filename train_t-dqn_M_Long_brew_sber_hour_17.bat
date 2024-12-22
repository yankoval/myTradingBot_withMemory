TITLE t-dqn_M_Long_brew_sber_hour_17
cd %USERPROFILE%
call env.bat 
rem tf\mytradingbot_withmemory
cd %USERPROFILE%\tf\mytradingbot_withmemory
python train.py --window-size=16 --batch-size=32 --episode-count=50 --model-name=t-dqn_M_Long_brew_sber_hour_17 --pretrained --tFrame=hourly --tik=SBER --dFrom=2017.01.30 --dTo=2017.05.15 --vdFrom=2018.03.01 --vdTo=2018.09.20
pause
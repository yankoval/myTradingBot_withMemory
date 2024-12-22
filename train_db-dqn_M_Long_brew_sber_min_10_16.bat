TITLE db-dqn_M_Long_brew_sber_min_10_16
cd %USERPROFILE%
call env.bat 
rem tf\mytradingbot_withmemory
cd %USERPROFILE%\tf\mytradingbot_withmemory
python train.py --window-size=10 --batch-size=16 --episode-count=50 --model-name=db-dqn_M_Long_brew_sber_min_10_16 --pretrained --tFrame=minute --tik=SBER --dFrom=2020.02.26 --tfCount=1000 --dTo=2020.03.03 --vdFrom=2020.03.03 --vdTo=2020.03.13
pause
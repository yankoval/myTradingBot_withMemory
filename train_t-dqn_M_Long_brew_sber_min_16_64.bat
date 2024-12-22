TITLE t-dqn_M_Long_brew_sber_min_16_64
REM cd %USERPROFILE%
pushd
call %USERPROFILE%\env.bat 
popd
cd
rem tf\mytradingbot_withmemory
REM cd %USERPROFILE%\tf\mytradingbot_withmemory
python train.py --window-size=16 --batch-size=64 --episode-count=50 --model-name=t-dqn_M_Long_brew_sber_min_16_64 --pretrained --tFrame=minute --tik=SBER --dFrom=2020.01.30 --tfCount=10000 --dTo=2020.05.15 --vdFrom=2018.03.01 --vdTo=2018.09.20
pause
echo off
SET window-size=10
SET batch-size=32
SET strategy=t-dqn
SET trStrat=short
SET tik=SBER
SET tFrame=hourly
SET episode-count=50
SET dFrom=2020.10.02 
SET dTo=2020.12.31 
SET vdFrom=2020.01.01 
SET vdTo=2020.03.31
SET pretrained=--pretrained

TITLE %strategy% %trStrat% %window-size% %batch-size% %tik% %tFrame% epCnt %episode-count% 
cd %USERPROFILE%
call env.bat 
rem tf\mytradingbot_withmemory
cd %USERPROFILE%\tf\mytradingbot_withmemory
python train.py --strategy=%strategy% --window-size=%window-size% --batch-size=%batch-size% --episode-count=%episode-count% ^
  --tFrame=%tFrame% --tik=%tik% --dFrom=%dFrom% --dTo=%dTo% --vdFrom=%vdFrom% --vdTo=%vdTo% --trStrat=%trStrat% %pretrained%
pause
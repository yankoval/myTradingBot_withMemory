echo off
SET window-size=16
SET batch-size=32
SET strategy=t-dqn
SET trStrat=short
SET tik=SBER
SET tFrame=minute
SET episode-count=50
SET dFrom=2018.02.01 
SET dTo=2018.02.02 
SET vdFrom=2018.03.01 
SET vdTo=2018.03.02
SET pretrained=
TITLE %strategy% %trStrat% %window-size% %batch-size% %tik% %tFrame% epCnt %episode-count% %pretrained%
cd %USERPROFILE%
call env.bat 
rem tf\mytradingbot_withmemory
cd %USERPROFILE%\tf\mytradingbot_withmemory
python train.py --strategy=%strategy% --window-size=%window-size% --batch-size=%batch-size% --episode-count=%episode-count%  ^
--tFrame=%tFrame% --tik=%tik% --dFrom=%dFrom% --dTo=%dTo% --vdFrom=%vdFrom% --vdTo=%vdTo% --trStrat=%trStrat% %pretrained%
pause
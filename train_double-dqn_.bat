rem echo OFF
SET window-size=14
SET batch-size=32

REM dqn double-dqn deff: t-dqn
SET strategy=double-dqn
SET trStrat=long
IF NOT [%1]==[]  (SET tik=%1) ELSE (SET tik=SBER)
SET tFrame=hourly
SET episode-count=100
SET tfCounts=10000
SET dFrom=2022.01.17
SET dTo=2023.04.01
SET vdFrom=2023.04.01
SET vdTo=2023.08.20
SET pretrained=
REM --pretrained
SET val-stock=
rem D:/share/finam/data/MXI-3.21/minute/fractalsValidate_MXI-3.21.csv
SET train-stock=
rem D:/share/finam/data/MXI-3.21/minute/MXI-3.21_fractal_2.csv
REM--pretrained  --debug
SET dataPath=Z:/finam/data/
SET debug=
SET rnd=%RANDOM%
IF NOT "%train-stock%"=="" (
SET r= train.py --trStrat=%trStrat% --tik=%tik% --strategy=%strategy% --window-size=%window-size% --batch-size=%batch-size% --episode-count=%episode-count% --tFrame=%tFrame% ^
--dFrom=%dFrom% --tfCounts=%tfCounts%  %pretrained% --dTo=%dTo% --vdFrom=%vdFrom% --vdTo=%vdTo% --train-stock=%train-stock% --val-stock=%val-stock%
--trainId=%rnd% --dataPath=%dataPath% %debug%
) ELSE (
SET r= train.py --trStrat=%trStrat% --tik=%tik% --strategy=%strategy% --window-size=%window-size% --batch-size=%batch-size% --episode-count=%episode-count%  ^
--tFrame=%tFrame% --dFrom=%dFrom% --dTo=%dTo% --vdFrom=%vdFrom% --vdTo=%vdTo% %pretrained% --tfCounts=%tfCounts% --trainId=%rnd% --dataPath=%dataPath%  %debug%
)


pushd
call venv\scripts\activate
popd
cd


echo %date% %time%: Start %rnd% %r% >> trainTasks.log
echo Starting
TITLE %date% %time%:%rnd%:%r:~10,200%
venv\scripts\python %r%
echo %date% %time%: Finish %rnd% %r% >> trainTasks.log

pause

rem window-size=16 batch-size=32 strategy=t-dqn trStrat=long tik=SBER tFrame=minute episode-count=50 tfCounts=2000 dFrom=2021.01.01 dTo=2021.01.30 vdFrom=2021.03.01 vdTo=2021.09.20 pretrained= val-stock= train-stock=
rem --trStrat=Long --tik=SBER --strategy=t-dqn --window-size=10 --batch-size=32 --episode-count=50  --tFrame=minute --dFrom=2021.01.01 --dTo=2021.01.30 --vdFrom=2021.02.01 --vdTo=2021.02.27 --tfCounts=2000
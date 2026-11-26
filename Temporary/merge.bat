@echo off
cd /d %~dp0
copy /b weights_part_00+weights_part_01+weights_part_02+weights_part_03+weights_part_04+weights_part_05+weights_part_06+weights_part_07+weights_part_08+weights_part_09 open_clip_pytorch_model.bin
copy /b transistor_part_00+transistor_part_01 transistor_test.zip
echo.
echo Merge done: open_clip_pytorch_model.bin and transistor_test.zip
pause

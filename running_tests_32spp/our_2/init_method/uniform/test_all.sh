#!/bin/bash

cd camera-obscura/
python3 ../../../../../scripts/test_analysis.py camera-obscura >> scores.txt
echo ''

cd ../dining-room/
python3 ../../../../../scripts/test_analysis.py dining-room >> scores.txt
echo ''

cd ../living-room/
python3 ../../../../../scripts/test_analysis.py living-room >> scores.txt
echo ''

cd ../modern-hall/
python3 ../../../../../scripts/test_analysis.py modern-hall >> scores.txt
echo ''

# cd ../modern-living-room/
# python3 ../../../scripts/test_analysis.py modern-living-room >> scores.txt
# echo ''


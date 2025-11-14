#!/bin/bash
# Скрипт компиляции C++ решения

clang++ -std=c++20 -O3 -march=native \
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 \
    -o solution_cpp solution.cpp

if [ $? -eq 0 ]; then
    echo "✓ Компиляция успешна: solution_cpp"
else
    echo "✗ Ошибка компиляции"
    exit 1
fi

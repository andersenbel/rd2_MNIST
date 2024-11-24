#!/bin/bash

# Тестування моделі A
python test.py --model A --model-path model_a.pth # >> results.txt
echo "Модель A протестована" >> results.txt

# Тестування моделі B
python test.py --model B --model-path model_b.pth #>> results.txt
echo "Модель B протестована" >> results.txt

# Тестування моделі C
python test.py --model C --model-path model_c.pth #>> results.txt
echo "Модель C протестована" >> results.txt

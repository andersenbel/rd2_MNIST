#!/bin/bash

# Навчання моделі A
python train.py --model A --output-path model_a.pth
echo "Модель A навчена та збережена в model_a.pth"

# Навчання моделі B
python train.py --model B --output-path model_b.pth
echo "Модель B навчена та збережена в model_b.pth"

# Навчання моделі C
python train.py --model C --output-path model_c.pth
echo "Модель C навчена та збережена в model_c.pth"

#!/bin/bash

# Скрипт для переименования и перекомпиляции отчетов лабораторных работ
# Переименовывает 0_main.tex в FM_labX_report.tex и компилирует

echo "Начинаем переименование и перекомпиляцию отчетов..."

for i in {1..6}; do
    echo "Обрабатываем лабораторную работу №$i..."
    
    # Проверяем существование папки
    if [ -d "lab$i" ]; then
        cd "lab$i"
        
        # Проверяем существование файла 0_main.tex
        if [ -f "0_main.tex" ]; then
            echo "  Переименовываем 0_main.tex в FM_lab${i}_report.tex..."
            mv 0_main.tex "FM_lab${i}_report.tex"
            
            echo "  Компилируем FM_lab${i}_report.tex..."
            xelatex "FM_lab${i}_report.tex" > /dev/null 2>&1
            
            # Проверяем успешность компиляции
            if [ -f "FM_lab${i}_report.pdf" ]; then
                echo "  ✓ Отчет FM_lab${i}_report.pdf успешно создан"
            else
                echo "  ✗ Ошибка при создании отчета для лабораторной работы №$i"
            fi
        else
            echo "  ⚠ Файл 0_main.tex не найден в папке lab$i"
        fi
        
        cd ..
    else
        echo "  ⚠ Папка lab$i не найдена"
    fi
    
    echo ""
done

echo "Переименование и перекомпиляция завершены!"
echo ""
echo "Созданные файлы:"
for i in {1..6}; do
    if [ -f "lab$i/FM_lab${i}_report.pdf" ]; then
        echo "  lab$i/FM_lab${i}_report.pdf"
    fi
done 
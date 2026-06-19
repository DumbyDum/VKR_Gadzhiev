# OmniGuard
Информация с детальным описанием собрана в пдф - файле (в описании написано "Текст дипломной работы")
Прототип для встраивания цифрового водяного знака в изображения, проверки payload/document_id, локализации изменений и сравнения исходного OmniGuard-подхода с улучшенной hybrid-версией.

## Возможности

- защита изображения цифровым водяным знаком;
- встраивание и проверка `document_id`;
- построение heatmap и бинарной маски изменений;
- сравнение изначального `watermark-only` метода и улучшенного `hybrid` метода;
- пакетный benchmark для нескольких изображений.

## Установка

```powershell
cd C:\Users\Mi\Downloads\OmniGuard-main\OmniGuard-main
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Запуск UI

```powershell
.\.venv\Scripts\python.exe app.py
```

Открыть в браузере:

```text
http://127.0.0.1:7860
```

Если порт занят:

```powershell
.\.venv\Scripts\python.exe app.py --port 7861
```

## Основные команды

Защитить изображение:

```powershell
.\.venv\Scripts\python.exe -m omniguard protect `
  --input examples\0000.png `
  --output runtime\artifacts\protected.png `
  --metadata runtime\artifacts\protected.json `
  --document-id demo-001
```

Проанализировать изображение:

```powershell
.\.venv\Scripts\python.exe -m omniguard analyze `
  --input runtime\artifacts\protected.png `
  --output-dir runtime\artifacts\analysis `
  --expected-document-id demo-001
```

Сравнить исходный и улучшенный методы:

```powershell
.\.venv\Scripts\python.exe -m omniguard paper-compare `
  --input-dir examples `
  --output-dir runtime\paper_comparisons\demo `
  --document-id paper-demo `
  --local-edit splicing_copy_move `
  --degradation gaussian_sigma10
```

Запустить benchmark:

```powershell
.\.venv\Scripts\python.exe -m omniguard benchmark `
  --input examples\0000.png `
  --output-dir runtime\benchmarks\demo `
  --document-id benchmark-001
```

## Форматы изображений

Поддерживаются: `PNG`, `JPG`, `JPEG`, `BMP`, `WEBP`.

Рекомендуется использовать изображения без сильного предварительного сжатия, желательно от `512x512` пикселей.

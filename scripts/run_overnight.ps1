Write-Host "=== OVERNIGHT TRAINING PIPELINE ===" -ForegroundColor Cyan
Write-Host "Start: $(Get-Date)" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Continue"

# 1. Vehicle Detector (YOLOv8m, 100 epochs, batch=2)
Write-Host "`n=== [1/4] Vehicle Detector ===" -ForegroundColor Green
python -m src.detection.train_vehicle_detector --epochs 100 --batch 2 --device 0 --no-prepare-labels
if ($?) {
    $src = "runs\detect\results\training\vehicle_detector\weights\best.pt"
    if (Test-Path $src) {
        Copy-Item $src "models\vehicle_detector.pt" -Force
        Write-Host "  Copied vehicle_detector.pt" -ForegroundColor Green
    }
}

# 2. Helmet Detector (YOLOv8s, 80 epochs, batch=4)
Write-Host "`n=== [2/4] Helmet Detector ===" -ForegroundColor Green
python -m src.detection.helmet_detector --epochs 80 --batch 4 --device 0 --no-prepare-labels
if ($?) {
    $src = "runs\detect\results\training\helmet_detector\weights\best.pt"
    if (Test-Path $src) {
        Copy-Item $src "models\helmet_detector.pt" -Force
        Write-Host "  Copied helmet_detector.pt" -ForegroundColor Green
    }
}

# 3. Triple Riding Detector (YOLOv8s, 80 epochs, batch=4)
Write-Host "`n=== [3/4] Triple Riding Detector ===" -ForegroundColor Green
python -m src.detection.triple_riding_detector --epochs 80 --batch 4 --device 0 --no-prepare-labels
if ($?) {
    $src = "runs\detect\results\training\triple_riding_detector\weights\best.pt"
    if (Test-Path $src) {
        Copy-Item $src "models\triple_riding_detector.pt" -Force
        Write-Host "  Copied triple_riding_detector.pt" -ForegroundColor Green
    }
}

# 4. Plate Detector (YOLOv8n, 80 epochs, batch=8)
Write-Host "`n=== [4/4] Plate Detector ===" -ForegroundColor Green
python -m src.ocr.train_plate_detector --epochs 80 --batch 8 --device 0 --no-prepare-labels
if ($?) {
    $src = "runs\detect\results\training\plate_detector\weights\best.pt"
    if (Test-Path $src) {
        Copy-Item $src "models\plate_detector.pt" -Force
        Write-Host "  Copied plate_detector.pt" -ForegroundColor Green
    }
}

Write-Host "`n=== DONE ===" -ForegroundColor Cyan
Write-Host "End: $(Get-Date)" -ForegroundColor Cyan
Get-ChildItem "models" -Filter "*.pt" | Select-Object Name, Length | Format-Table -AutoSize

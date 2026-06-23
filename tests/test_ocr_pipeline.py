from src.ocr.ocr_pipeline import validate_plate


def test_validate_plate_accepts_valid_indian_plate():
    assert validate_plate("KA01AB1234") is True


def test_validate_plate_rejects_invalid_plates():
    assert validate_plate("") is False
    assert validate_plate("INVALID") is False
    assert validate_plate(None) is False

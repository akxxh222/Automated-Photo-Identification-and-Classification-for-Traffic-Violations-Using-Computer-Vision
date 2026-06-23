import pytest
from unittest.mock import MagicMock, patch

from src.database.database import get_db


def test_get_db_rolls_back_on_exception():
    mock_db = MagicMock()
    with patch("src.database.database.SessionLocal", return_value=mock_db):
        gen = get_db()
        next(gen)
        with pytest.raises(RuntimeError):
            gen.throw(RuntimeError("simulated db error"))
        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()

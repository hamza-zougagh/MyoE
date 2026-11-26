from pathlib import Path
import pytest

streamlit = pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest


def test_synthetic_demo_renders_without_exception():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'demo' / 'app.py', default_timeout=45).run()
    assert not app.exception
    assert not app.error
    assert app.title[0].value == 'MyoE'
    assert any(button.label == 'Personalize model' for button in app.button)

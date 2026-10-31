"""Entry point: python app.py"""

import os

from medication_app import create_app
from medication_app.config import SERVER_PORT

app = create_app()
server = app.server  # for a WSGI host (gunicorn/waitress)

if __name__ == "__main__":
    # Public run: debug off (no dev bar). For development: MEDAPP_DEBUG=1 python app.py
    app.run(debug=os.environ.get("MEDAPP_DEBUG") == "1", port=SERVER_PORT)

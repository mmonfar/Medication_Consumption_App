"""Entry point: python app.py"""

from medication_app import create_app
from medication_app.config import SERVER_PORT

app = create_app()
server = app.server  # for a WSGI host (gunicorn/waitress)

if __name__ == "__main__":
    app.run(debug=True, port=SERVER_PORT)

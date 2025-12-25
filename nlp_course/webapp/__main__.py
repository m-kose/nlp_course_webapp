from __future__ import annotations

import os

from .app import create_app


def main() -> None:
    app = create_app()
    host = os.environ.get("WEBAPP_HOST", "127.0.0.1")
    port = int(os.environ.get("WEBAPP_PORT", "5000"))
    debug = os.environ.get("WEBAPP_DEBUG", "0").lower() in {"1", "true", "yes", "y"}
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()


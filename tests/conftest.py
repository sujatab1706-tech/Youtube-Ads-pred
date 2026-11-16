import os
import tempfile

# Keep test predictions out of the real logs/ folder. Must run before app.main is imported.
os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="ads-test-logs-"))

import logging
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
test_samples_dir = test_dir/'samples'

log_file = test_dir/'test.log'
if log_file.exists():
    log_file.unlink()
logging.basicConfig(filename=log_file,level=logging.DEBUG)

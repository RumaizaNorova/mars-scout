# Python locks

Edit a `.in` file, then regenerate its `.txt`. Never install from a `.in` file.

```bash
uv pip compile requirements/ros.in --python-version 3.12 --python-platform linux -o requirements/ros.txt
uv pip compile requirements/telemetry.in --python-version 3.12 --python-platform linux -o requirements/telemetry.txt
uv pip compile requirements/test.in --python-version 3.12 --python-platform linux -o requirements/test.txt
uv pip compile requirements/isaac.in --python-version 3.11 --python-platform linux --no-emit-package numpy -o requirements/isaac.txt
```

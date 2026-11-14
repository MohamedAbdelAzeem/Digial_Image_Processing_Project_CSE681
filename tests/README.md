# Tests

Run the tests from the repository root with the `dip-project1` Conda environment active.

To run the ScaleHyperprior integration tests:

```powershell
python -m pytest tests/test_integration_hyperprior.py -q
```

## Options

| Option | Meaning |
| --- | --- |
| `-q` | Quiet output: dots and a one-line summary instead of one line per test. |
| `-k <expression>` | Run only tests whose names match, e.g. `-k tiny_training`. |
| `-o log_cli=true` | Print log messages live while tests run; by default pytest shows them only for failing tests. |
| `--log-cli-level=INFO` | Lowest level shown live. Training metrics are logged at `INFO`, so `WARNING` (the default) hides them. |
| `-o log_file=test.log` | Also write logs to this file in the current directory; do not commit it. |
| `-o log_file_level=INFO` | Lowest level written to the log file. |

Example, showing the training logs live:

```powershell
python -m pytest tests/test_integration_hyperprior.py -k tiny_training -o log_cli=true --log-cli-level=INFO
```

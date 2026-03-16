# Valere

LLM-powered experiment runner.

## Configuration

Valere loads configuration from multiple sources with the following precedence (highest to lowest):

1. Command-line arguments
2. Environment variables
3. Configuration file
4. Built-in defaults

### Configuration File

Copy `valere.toml.example` to `valere.toml` and customize:

```bash
cp valere.toml.example valere.toml
```

See `valere.toml.example` for all available options.

### Environment Variables

All CLI flags can be set via environment variables:

| CLI Flag | Environment Variable |
|----------|---------------------|
| `--dry-run` | `VALERE_DRY_RUN` |
| `--validate` | `VALERE_VALIDATE` |
| `--config` | `VALERE_CONFIG` |
| `--embedding-model` | `VALERE_EMBEDDING_MODEL` |
| `--experiment-model` | `VALERE_EXPERIMENT_MODEL` |
| `--hypothesis-model` | `VALERE_HYPOTHESIS_MODEL` |
| `--max-runs` | `VALERE_MAX_RUNS` |
| `--max-seconds` | `VALERE_MAX_SECONDS` |

### Command-Line Options

```
valere --help
```

## Validation

Validate your configuration without running the experiment:

```bash
valere --validate
```

This checks that:
- Model names are non-empty
- Numeric values are within valid ranges
- Required prompt files exist

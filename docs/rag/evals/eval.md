# Evaluation

We make use of the [Weave Evaluation API](https://weave-docs.wandb.ai/guides/core-types/evaluations) to evaluate the performance of different stochastic components in the pipeline.

Evaluations can be run using the CLI `medrag evaluate`.

```
▶ medrag evaluate --help
usage: medrag evaluate [-h] [--test-file TEST_FILE] [--test-case TEST_CASE]

options:
  -h, --help            show this help message and exit
  --test-file TEST_FILE
                        Path to test file
  --test-case TEST_CASE
                        Only run tests which match the given substring expression
  --model-name MODEL_NAME
                        Model name to use for evaluation
```
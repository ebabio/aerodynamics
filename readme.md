# Lifting Line

## What to do
Structure of the code:
1. A discretizer of the span
2. A numerical sovler for the integral equation, it could be gradient based or we can try the proposed fixed point iteration
3. The lifting line model that we can modify in the future

## Pixi
In this project I'm trying for Pixi for the first time, the goal is to use to substitute `conda`, `pypi` into a single environment, and also use to describe the package installation substituting the `pyproject.toml`.

First of all, some useful references:
* Pixi installation and introduction: https://pixi.sh/v0.47.0/
* Integrating into VSCode: https://pixi.sh/dev/integration/editor/vscode/
    * Requires an available environment first: run `pixi install --environment $envname$` in the terminal, where `$envname$` will typically be `test`
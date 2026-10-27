# Contributing to BaseAL

Thanks for your interest in BaseAL. This page explains how to get help, how to report a problem, and how to contribute code or documentation.

## Getting support

- **Questions about using BaseAL**: first check the [notebooks in `core/docs/`](core/docs/) and the [API reference](https://benmcewen1.github.io/BaseAL/). If they don't answer it, [open an issue](https://github.com/BenMcEwen1/BaseAL/issues/new/choose) using the **Question** template.
- **Private enquiries** (for example collaboration, or for questions not applicable to other users of BaseAL): email [Ben McEwen](mailto:benmcewen@outlook.com).

## Reporting a problem

Before opening an issue, search the [existing issues](https://github.com/BenMcEwen1/BaseAL/issues?q=is%3Aissue) in case it has already been reported.

Then [open a new issue](https://github.com/BenMcEwen1/BaseAL/issues/new/choose) using the **Bug report** template, and include:

- What you ran (code, notebook cell or command) and what you expected to happen
- The full error message or traceback
- Your operating system, Python version, and how you installed BaseAL (`uv sync`, conda, or the Docker/Apptainer image)
- The BaseAL version (`version` in [pyproject.toml](pyproject.toml)) or the commit you're on
- The dataset you used, if the problem depends on it

For problems with the web app, also include your Node.js version (`node --version`) and any errors from the browser console.

## Contributing code or documentation

Bug fixes, new sampling strategies, documentation and example notebooks are all welcome. For a larger change, such as a new feature or a refactor, open an issue first so we can agree on the approach before you put time into it.

### 1. Set up a development copy

1. [Fork the repository](https://github.com/BenMcEwen1/BaseAL/fork) and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/BaseAL.git
   cd BaseAL
   ```

2. Install the dependencies, including the test and docs extras. See [Prerequisites](README.md#prerequisites) for the tools you need.

   ```bash
   uv sync --extra test --extra docs
   ```

3. Create a branch for your change:

   ```bash
   git checkout -b my-change
   ```

### 2. Make your change

- **Follow the existing style.** Write Google-style docstrings (`Args:`, `Returns:`, `Raises:`) for public classes and functions; the [API reference](https://benmcewen1.github.io/BaseAL/) is generated from them.
- **Add tests.** New behaviour should come with tests in [`tests/`](tests/). The tests use small synthetic datasets built in [`tests/conftest.py`](tests/conftest.py), so no data download is needed.
- **New sampling strategies** go in [`core/utils/sampling.py`](core/utils/sampling.py). Add a test to [`tests/test_sampling.py`](tests/test_sampling.py), and ideally an example to the notebooks.
- **Web app changes** (in `app/`) should build cleanly with `npm run build`.

### 3. Check your change

Run the test suite:

```bash
uv run pytest -v
```

If you changed docstrings, check that the API reference builds and renders as you expect:

```bash
uv run pdoc core core.active_learner core.utils api.main -d google
```

This opens a live preview in your browser.

### 4. Open a pull request

Push your branch to your fork and [open a pull request](https://github.com/BenMcEwen1/BaseAL/compare) against `main`. In the description, explain what the change does and why, and link any related issue (for example `Fixes #12`).

GitHub Actions runs the test suite on every pull request. Please make sure it passes; a maintainer will then review your change.

## Authorship and credit

Authors of substantial contributions will be recognised as authors in the citable DOI for subsequent releases. Substantial contributions include:

- Implementing a major feature or new functionality
- Significant refactoring or performance improvements
- Comprehensive documentation or tutorials
- Bug fixes that resolve critical issues

Minor contributions (small bug fixes, typos, formatting) will be acknowledged in the release notes.

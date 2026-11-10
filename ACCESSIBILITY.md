# Accessibility

`obscure_stats` is a Python library: it has no graphical interface, and people use it through code, documentation, and
messages in their terminal or notebook. We want everyone, including people with disabilities and people using assistive
technology, to be able to read the documentation, use the library, report problems, and contribute.

This document explains what accessibility means for this project, how contributors can help keep it accessible, and how to
report a barrier.

## Priorities

- **Readable documentation.** The README, `CONTRIBUTING.md`, `CHANGELOG.md` and other Markdown files use plain language,
  a logical heading hierarchy, real lists, descriptive link text, and meaningful alternative text for images and badges.
- **Text, not images.** Formulas and examples in docstrings and documentation are written as plain text, so they can be
  read by screen readers, copied, and searched, and formulas are also described in words.
- **Clear messages.** Errors and warnings say what is wrong and what to do about it (for example,
  "Parameter k should be in range [0, 0.5)."), and do not rely on color or formatting to convey meaning.
- **Predictable results.** Functions return a number or NaN with a warning, rather than failing silently, so problems are
  noticeable without inspecting output visually.
- **Accessible contribution.** Issues, pull requests and reviews happen on GitHub; we do not require tools or formats
  beyond what GitHub and a text editor provide.

We follow the principles of [WCAG 2.2](https://www.w3.org/TR/WCAG22) for our Markdown documentation as an aspirational
goal. This is not a claim of verified conformance.

## Contributor expectations

If you contribute code or documentation, please follow these guidelines so we do not introduce new barriers:

- **Documentation and Markdown**
  - Use a logical heading hierarchy (do not skip levels).
  - Use real lists and tables with header rows, not text formatted to look like them.
  - Use unique, descriptive link text (avoid "click here" or "read more").
  - Give every image and badge meaningful alternative text; see the
    [W3C alt decision tree](https://www.w3.org/WAI/tutorials/images/decision-tree/).
  - Do not use color, emoji, or formatting as the only way to convey meaning.
- **Docstrings**
  - Follow the existing `numpy` style: start with a verb, describe the inputs and the returned value in words.
  - Write formulas in plain text (for example, `(mean - median) / mean(|x - median|)`), not as images, and describe
    them in words next to the notation (for example, "the mean minus the median, divided by the mean absolute
    deviation from the median").
  - Add a short example with its output. Examples use the standard `>>>` prompt so that they can be run as
    tests, which keeps the shown output correct.
- **Errors and warnings**
  - Say what went wrong and how to fix it, in a full sentence.
  - Name the parameter or input involved.

We do not currently run automated accessibility checks in CI. Reviewers check documentation changes against the list above.

## Reporting accessibility issues

If something in this project is hard or impossible for you to use, please tell us. We treat accessibility reports as
valuable expertise, not complaints.

1. **Open an issue** at <https://github.com/glevv/obscure_stats/issues>, or, if you prefer not to post publicly, email
   <securityobscurestats.cer9a@slmail.me>.
2. If you can, include:
   - What you were trying to do and what went wrong.
   - Where it happened (for example, the README, a function's docstring, an error message, or the PyPI page).
   - The tools you use (for example, your screen reader, browser, editor or IDE, and their versions).
   - A screenshot or recording, only if you are comfortable sharing one.

You do not need to disclose a disability or any other personal information.

### Severity

Maintainers assign severity during triage; reporters do not need to.

- **Critical:** prevents you from using the library or its documentation for a core task (for example, a function's
  documentation cannot be read at all).
- **Serious:** significant difficulty, but a workaround exists.
- **Moderate:** an annoyance or inconsistent experience.
- **Minor:** minimal impact on use.

### How we respond

- We will acknowledge your report within 14 days.
- We will explain next steps, known limitations, and any dependencies outside this project.
- Where possible, we will suggest a workaround while a fix is in progress.
- We will post updates on the issue when the status or expected timeline changes.
- We may ask you to confirm that a fix removes the barrier before closing the issue.

### Resolution expectations

`obscure_stats` is maintained by volunteers, so we cannot promise fixed deadlines. Accessibility issues are prioritised
above new features, in order of severity, and each issue will show its status and, once known, a target release.

## Ownership and maintenance

Accessibility is owned by the project maintainer, [@glevv](https://github.com/glevv), who is responsible for:

- triaging accessibility reports;
- tracking known barriers in GitHub issues;
- keeping this document up to date.

## Supported environments

This project is used through:

- **Documentation on GitHub and PyPI**, rendered in a web browser.
- **Docstrings**, read through Python's `help()`, IDEs, notebooks, or the source code.
- **Messages** (errors and warnings) in terminals and notebooks.

We have not tested these with specific assistive technologies; please let us know about combinations that do not work.

## Known limitations

- There is no rendered math: formulas in docstrings are written in plain-text notation, each with a description in
  words next to it. Rendered, screen-reader-friendly math would need a documentation website, which the project does
  not have.
- There is no separate documentation website: documentation lives in the README and in docstrings.
- Documentation is available in English only.

Please open an issue if you find a barrier that is not listed here.

## Feedback

Accessibility is an ongoing practice. If you have suggestions for improving this document or our practices, please open an
issue or a pull request.

This document is adapted from the [ACCESSIBILITY.md template](https://github.com/open-source-accessibility/accessibility-toolkit/blob/main/ACCESSIBILITY.md)
of the Accessibility Toolkit (MIT License).

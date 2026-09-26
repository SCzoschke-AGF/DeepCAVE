# Proportional execution

Keep the amount of investigation and verification proportional to the requested change.

For small, local changes such as moving or renaming one configuration option:

1. Inspect only the directly affected configuration, code reference, and test reference.
2. Make one focused patch that changes only those locations.
3. Run only the smallest relevant tests. Do not run the full suite unless the change is broad, high-risk, or targeted tests reveal wider impact.
4. Do not investigate unrelated failures or modify unrelated files. Report them briefly if encountered.
5. Stop once the requested behavior and targeted tests are verified.

Avoid repeated defensive checks when the current file contents already establish the necessary facts. Do not browse the web for purely local edits unless an external API detail must be verified.

If `apply_patch` fails with the known `error building bubblewrap command: mountinfo path is not absolute` infrastructure error, do not retry it and do not experiment with multiple patch utilities. Use `uv run --locked python -` as the single fallback; plain `python` is not available in this environment.

For that fallback, use exact-match replacements with uniqueness checks. Read and validate every intended replacement first, build all updated file contents in memory, and write files only after every check succeeds, so a later mismatch cannot leave partial edits. Then run `git diff --check` and only the targeted tests. Remove any `.orig` or `.rej` files created by patch tooling.

For interrupted commands, check for a lingering process only when the command could realistically still be running.

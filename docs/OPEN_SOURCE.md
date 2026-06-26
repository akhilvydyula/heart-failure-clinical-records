# Open Source Inventory

Open-source features enabled for **Heart Failure Clinical Records**.

## Repository

| Item | Details |
|------|---------|
| License | [MIT](../LICENSE) |
| Visibility | Public on GitHub |
| Maintainer | [Akhil Vydyula](https://github.com/akhilvydyula) |
| Tech stack | Python, scikit-learn, DuckDB, FastAPI, Streamlit, Invoke |

## Community

| Resource | Link |
|----------|------|
| Contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Code of conduct | [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) |
| Bug reports | [Issue template](../.github/ISSUE_TEMPLATE/bug_report.yml) |
| Feature requests | [Issue template](../.github/ISSUE_TEMPLATE/feature_request.yml) |
| Pull requests | [PR template](../.github/pull_request_template.md) |
| Discussions | [GitHub Discussions](https://github.com/akhilvydyula/heart-failure-clinical-records/discussions) |

## Security and compliance

| Resource | Link |
|----------|------|
| Security policy | [SECURITY.md](../SECURITY.md) |
| Compliance | [COMPLIANCE.md](COMPLIANCE.md) |
| Dependabot | [`.github/dependabot.yml`](../.github/dependabot.yml) |

## CI/CD (GitHub Actions)

| Workflow | Purpose |
|----------|---------|
| [ci.yml](../.github/workflows/ci.yml) | Tests, lint, `pip-audit`, Bandit SAST |
| [codeql.yml](../.github/workflows/codeql.yml) | CodeQL semantic security analysis |

```text
push / pull_request -> main
  |- test (syntax check or pytest)
  |- dependency-audit (pip-audit)
  `- sast-bandit (Bandit)
```

## Get started

1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Star the repository
3. Open or pick an [issue](https://github.com/akhilvydyula/heart-failure-clinical-records/issues)
4. Fork, branch, and open a PR

Thank you for supporting open-source ML education!

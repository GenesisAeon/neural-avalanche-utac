# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-06-17
### Added
- Initial v1.0.0 release as part of the GenesisAeon ecosystem-wide 1.0.0
  milestone.
- `RELEASE_GUIDE.md`, `CONTRIBUTING.md`, issue/PR templates.

### Changed
- Project metadata (`pyproject.toml`, `CITATION.cff`) bumped to 1.0.0.
- `.zenodo.json` updated to 1.0.0 and corrected to MIT licensing to match
  this repo's actual `LICENSE` file (previously listed as
  `GPL-3.0-or-later`, which did not match the code license).
- `.github/workflows/release.yml` extended with a test job and a
  canary (TestPyPI / pre-release) publishing channel for `-rc`/`-alpha`/
  `-beta` tags.

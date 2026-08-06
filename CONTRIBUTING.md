# Contribution Guidelines

Thank you for contributing to the **Vision-Based Robotic Arm** project. To maintain code quality, clear version history, and smooth team collaboration, all contributors must adhere to the guidelines below.

---

## Team Workflow & Collaboration Rules

### 1. Own GitHub Accounts
- Each team member must commit and submit pull requests using their **own personal GitHub account**.
- Shared accounts or committer impersonation are prohibited.

### 2. Feature Branch Strategy
- **Direct pushes to `main` branch are strictly prohibited.**
- All development must take place on dedicated feature or bugfix branches created off `main`.
- Naming convention for branches:
  - `feature/<member>-<feature-description>` (e.g., `feature/m1-dataset-loader`)
  - `docs/<member>-<doc-name>` (e.g., `docs/m2-architecture`)
  - `fix/<member>-<bug-description>` (e.g., `fix/m3-gitignore-update`)

### 3. Meaningful Commit Messages
Commit messages should be concise, imperative, and descriptive:
- Good: `feat(dataset): add recursive image scanner with CLI args`
- Good: `docs(architecture): document confidence-gated perception pipeline`
- Bad: `updated stuff`, `fix`, `changes`

### 4. Pull Request (PR) & Code Review Process
- Before merging into `main`, open a Pull Request (PR) against `main`.
- Provide a clear summary of changes in the PR description.
- Every PR requires **at least one review and explicit approval** from another team member before merging.
- Ensure all automated checks and verification commands pass before requesting merge.

---

## Development Environment Setup

1. Clone repository:
   ```bash
   git clone https://github.com/<your-username>/vision-based-robotic-arm.git
   cd vision-based-robotic-arm
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   ```

3. Activate environment:
   - Windows: `.venv\Scripts\activate`
   - Linux/macOS: `source .venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Verifying Code Before Opening a PR

Run available starter tests prior to submitting changes:

```bash
# Verify dataset loading utility
python data/scripts/load_dataset.py --data-dir data/sample

# Verify PyBullet simulation scene
python simulation/scene.py
```

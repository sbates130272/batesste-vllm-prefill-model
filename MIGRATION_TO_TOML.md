# Migration to pyproject.toml

## ✅ What Was Created

### New Files:
1. **`pyproject.toml`** - Modern Python project configuration
   - Project metadata (name, version, description, authors)
   - Production dependencies
   - Optional dev dependencies (`[dev]`)
   - Optional ML dependencies (`[ml]`)
   - pytest configuration
   - black configuration
   - isort configuration

2. **`.flake8`** - Flake8 configuration
   - (flake8 doesn't support pyproject.toml yet)
   - Already starts with dot (hidden file)

### Updated Files:
- **`.github/workflows/ci.yml`** - Updated to use `pip install -e ".[dev]"`

---

## 🗑️ Files That Can Be Removed

Once you verify everything works:

1. **`requirements.txt`** ❌ Can remove
2. **`requirements-dev.txt`** ❌ Can remove
3. **`pytest.ini`** ❌ Can remove (config moved to pyproject.toml)
4. **`setup.py`** ❌ Can remove (replaced by pyproject.toml)

---

## 📦 New Installation Methods

### For Production/Running:
```bash
pip install -e .
```

### For Development (includes test/lint tools):
```bash
pip install -e ".[dev]"
```

### For ML workloads (includes PyTorch):
```bash
pip install -e ".[ml]"
```

### Everything:
```bash
pip install -e ".[all]"
```

---

## 🔄 Update Your Scripts

### `setup-venv.sh`:
Change from:
```bash
pip install -r requirements.txt
```
To:
```bash
pip install -e .
```

### `restart-webui.sh`:
Dependency checks can remain as-is (they check imports)

---

## ✅ Testing the Migration

### Step 1: Test in a fresh venv
```bash
# Create test venv
python3 -m venv test-venv
source test-venv/bin/activate

# Install from pyproject.toml
pip install -e ".[dev]"

# Verify dependencies
python3 -c "import fastapi, uvicorn, pydantic, transformers"
python3 -c "import pytest, black, isort, flake8"

# Run tests
pytest tests/ -v

# Cleanup
deactivate
rm -rf test-venv
```

### Step 2: Verify CI still works
```bash
# Commit and push changes
git add pyproject.toml .flake8 .github/workflows/ci.yml
git commit -m "build: migrate to pyproject.toml for modern Python packaging"
git push
```

### Step 3: Remove old files
```bash
git rm requirements.txt requirements-dev.txt pytest.ini setup.py
git commit -m "build: remove legacy configuration files"
git push
```

---

## 📝 Commit Plan

### Commit 1: Add pyproject.toml
```bash
git add pyproject.toml .flake8 .github/workflows/ci.yml
```

**Message:**
```
build: migrate to pyproject.toml for modern Python packaging

- Add pyproject.toml with project metadata and dependencies
- Consolidate pytest, black, and isort configuration
- Add .flake8 config (flake8 doesn't support pyproject.toml)
- Update CI workflow to use pip install -e ".[dev]"
- Define optional dependency groups: dev, ml, all

Benefits:
- Single source of truth for project configuration
- Modern Python packaging standard (PEP 517/518)
- Easier dependency management
- Cleaner root directory (when old files removed)

Next step: Remove requirements.txt, requirements-dev.txt,
pytest.ini, and setup.py after verification.
```

### Commit 2: Remove legacy files (after testing)
```bash
git rm requirements.txt requirements-dev.txt pytest.ini setup.py
```

**Message:**
```
build: remove legacy configuration files

Remove deprecated configuration files now that pyproject.toml
is in place:
- requirements.txt (moved to pyproject.toml)
- requirements-dev.txt (moved to [project.optional-dependencies])
- pytest.ini (moved to [tool.pytest.ini_options])
- setup.py (replaced by pyproject.toml build-system)

All functionality preserved in pyproject.toml.
```

---

## 🎯 Benefits

1. **Single Configuration File**: Most config in one place
2. **Modern Standard**: PEP 517/518 compliant
3. **Better Dependency Management**: Optional dependency groups
4. **Cleaner Root**: Fewer config files
5. **Tool Support**: Black, isort, pytest all support pyproject.toml

---

## 🚨 Important Notes

- Keep `requirements.txt` temporarily until you've tested
- CI will use the new approach automatically
- Local development: remember to use `pip install -e ".[dev]"`
- Documentation should be updated to reflect new install method

---

## ✅ Ready to Proceed?

1. Review `pyproject.toml` - check metadata (author, URLs, etc.)
2. Test installation in a clean venv
3. Commit the new configuration
4. Verify CI passes
5. Remove old files

This is a non-breaking change - both approaches can coexist during migration.


# f90wrap Issues and PRs Triage Plan

**Date:** December 14, 2025
**Branch:** `claude/triage-issues-and-prs-O7A4w`

## Executive Summary

There are **5 open PRs** ready for review that fix specific issues, plus **1 version bump PR**. Several of these PRs are from active contributors (Rykath, krystophny) and have associated test cases. Additionally, there are **~20 open issues** ranging from recent bug reports to long-standing feature requests.

---

## Priority 1: PRs Ready for Review (Easiest to Resolve)

These PRs have associated issues, include test cases, and are marked as "clean" (mergeable):

| PR | Issue | Title | Author | Files | Status |
|----|-------|-------|--------|-------|--------|
| #298 | #297 | Ignore unknown parent derived types | Rykath | 8 | ✅ Clean, 3 commits |
| #300 | #299 | Fix direct-c nested function definitions | krystophny | 11 | ✅ Clean, 5 commits |
| #303 | #301 | Add default complex type mappings | krystophny | 11 | ✅ Clean, 4 commits |
| #304 | #302 | Warn about pointer arrays in derived types | krystophny | 12 | ✅ Clean, 6 commits |
| #308 | #305 | Fix multiple abstract interfaces | Rykath | 7 | ✅ Clean, 2 commits |

### Detailed Code Review:

#### 1. **PR #298** - Backwards compatibility fix for unknown parent derived types
- **File:** `f90wrap/transform.py` (3 lines added)
- **Change:** In `find_inheritence_relations()`, adds a check before accessing `type_map[parent]`:
  ```python
  if parent not in type_map:
      log.warning("Parent type %s of type %s not found" % (parent, node.name))
      continue
  ```
- **Risk:** ✅ Very Low - graceful degradation with warning
- **Test:** `examples/issue297_ignored_abstract_classes/`
- **Recommendation:** **MERGE** - Simple, safe fix that restores backwards compatibility

#### 2. **PR #300** - Fix C compilation error in direct-c mode for module arrays
- **File:** `f90wrap/directc_cgen/module_helpers.py` (3 lines added)
- **Change:** Adds missing function closure after `write_array_helper_body()`:
  ```python
  gen.dedent()
  gen.write("}")
  gen.write("")
  ```
- **Risk:** ✅ Very Low - fixes obvious bug (missing closing brace)
- **Test:** `examples/issue299_directc_nested_functions/`
- **Recommendation:** **MERGE** - Clear bug fix for nested function definition error

#### 3. **PR #303** - Add `complex` and `double complex` to default type mappings
- **File:** `f90wrap/fortran.py` (2 lines added)
- **Change:** Extends `default_f2c_type` dictionary:
  ```python
  'complex': 'complex_float',
  'double complex': 'complex_double',
  ```
- **Risk:** ✅ Very Low - extends existing functionality
- **Test:** `examples/issue301_complex_types/`
- **Recommendation:** **MERGE** - Simple extension that eliminates RuntimeError

#### 4. **PR #304** - Add warning when pointer arrays are skipped in direct-c mode
- **File:** `f90wrap/directc_cgen/__init__.py` (9 lines added)
- **Change:** Before processing derived type elements, checks for pointer arrays:
  ```python
  is_pointer = "pointer" in element.attributes
  if is_pointer and is_array:
      print(f"f90wrap: skipping {derived.name}.{element.name} "
            "(pointer arrays not supported in direct-c mode)", file=sys.stderr)
      continue
  ```
- **Risk:** ✅ Very Low - only adds warning, no behavior change
- **Test:** `examples/issue302_pointer_warning/`
- **Recommendation:** **MERGE** - Improves user experience with clear warning

#### 5. **PR #308** - Skip Python wrapper generation for abstract interfaces
- **File:** `f90wrap/pywrapgen.py` (4 lines added)
- **Change:** In `visit_Interface()`, skip abstract interfaces:
  ```python
  if "abstract" in node.attributes:
      log.info(" -> abstract interface, skipping")
      return
  ```
- **Risk:** ✅ Low - abstract interfaces can't be instantiated anyway
- **Test:** `examples/issue305_multiple_abstract_interfaces/`
- **Recommendation:** **MERGE** - Prevents generation of unusable wrapper code

---

## Priority 2: Version Bump PR

| PR | Title | Author | Status |
|----|-------|--------|--------|
| #288 | Bump version to 0.3.0 | jameskermode | ✅ Clean, single file |

### Recommended Actions:
- **Wait** until Priority 1 PRs are merged
- Consider if #287 (Prep for v0.3.0 release) requirements are met
- Merge after all v0.3.0 features are included

---

## Priority 3: Recent Issues Without PRs

### Issue #307 - FIXED (in this branch)

**Problem:** Logical arrays require int32 workaround - misleading docstring says bool

**Root Cause:**
- Fortran `logical` is 4 bytes (same as `integer`)
- f2py maps it to C `int`
- NumPy `bool` is only 1 byte
- The docstring incorrectly said `bool array` when users must pass `int32 array`

**Fix Applied:**
1. `f90wrap/pywrapgen.py` - In `_format_pytype()`, map "bool array" → "int32 array" for docstrings
2. `f90wrap/numpy_utils.py` - In `numpy_type_from_fortran()`, map logical → `NPY_INT32` (not `NPY_BOOL`)
3. `test/test_directc.py` - Updated test to expect `NPY_INT32` for logical
4. Added test case: `examples/issue307_logical_array/`

### Remaining Issues (Moderate Complexity)

| Issue | Title | Complexity | Notes |
|-------|-------|------------|-------|
| #306 | Module-level allocatable arrays fail after reallocation | Medium-High | Dynamic array dimension tracking issue. Wrapper caches original size |

### Lower Priority (Documentation/CI)

| Issue | Title | Status |
|-------|-------|--------|
| #293 | Documentation CI not available | Has docs, needs CI setup. Contributor offered to help |
| #287 | Prep for v0.3.0 release | Waiting on PRs to merge |

---

## Priority 4: Long-Standing Issues

These require more investigation or are feature requests:

| Issue | Title | Age | Category |
|-------|-------|-----|----------|
| #269 | `--py-mod-names` behavior | 4 months | Clarification needed |
| #253 | Module parameter not imported for argument kind | 9 months | Type handling |
| #226 | f2py-f90wrap v0.2.15 causes .mod error | 1+ year | Compatibility |
| #223 | Why is FortranModule a Singleton? | 1+ year | Design question |
| #222 | Handle collision in array wrapper breaks CI | 1+ year | Intermittent CI |
| #219 | Wrapping code dependencies & ordering | 1+ year | Documentation |
| #204 | Callback function example fails | 2+ years | Callback support |
| #196 | Can f90wrap parse FORD docstrings? | 2+ years | Feature request |
| #195 | Routines with procedure arguments | 2+ years | Procedure pointers |
| #187 | Issues on Mac OS X for f90wrap 0.2.12 | 2+ years | Platform-specific |
| #171 | `integer, value` arguments converted to real | 3+ years | Type handling |
| #170 | Allocatable character arrays in derived types | 3+ years | Feature request |

---

## Recommended Merge Order

```
1. PR #298 (backwards compat - low risk)
   ↓
2. PR #300 (direct-c fix - low risk)
   ↓
3. PR #303 (complex types - extends functionality)
   ↓
4. PR #304 (warning only - very low risk)
   ↓
5. PR #308 (abstract interfaces - review carefully)
   ↓
6. PR #288 (version bump to 0.3.0)
```

---

## Quick Wins for New Contributors

1. **Issue #293** - Set up documentation CI (ReadTheDocs or GitHub Pages)
2. **Issue #307** - Fix misleading docstring for logical arrays (documentation fix)
3. Improve error messages based on #301, #302 patterns

---

## Files Most Relevant to Open Issues

| File | Related Issues |
|------|----------------|
| `f90wrap/fortran.py` | #301, #303 (type mappings) |
| `f90wrap/directc_cgen/module_helpers.py` | #299, #300 (direct-c generation) |
| `f90wrap/transform.py` | #297, #298 (inheritance tree) |
| `f90wrap/pywrapgen.py` | #305, #308 (Python wrapper generation) |
| `f90wrap/numpy_utils.py` | #306, #307 (array handling) |

---

## Next Steps

1. Run CI on all open PRs to verify tests pass
2. Review PRs in recommended order
3. Merge PRs that pass review
4. Create v0.3.0 release after merging
5. Address #307 and #306 as follow-up issues

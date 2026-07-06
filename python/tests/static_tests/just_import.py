"""
NEST GPU Test: Module Import Validation
========================================

This test validates that the NEST GPU module can be successfully imported.
It is the most basic sanity check for the installation.

Purpose:
-------
- Verify NEST GPU is properly installed
- Check Python bindings are working
- Validate library path configuration
- Test basic module availability

Test Procedure:
--------------
1. Attempt to import nestgpu module
2. If successful, test passes (exit code 0)
3. If import fails, test fails (exit code non-zero)

Expected Result:
---------------
- Clean import with no errors
- No missing dependencies
- Library path correctly configured

Common Issues:
--------------
- NESTGPU_LIB environment variable not set
- CUDA library not found
- Python version incompatibility
- Library path misconfiguration

Usage:
------
    python just_import.py

Or via test runner:
    pytest just_import.py

Dependencies:
-------------
- Python 3.x
- NEST GPU library
- CUDA toolkit
- Proper environment configuration

Return Value:
------------
- 0: Success (module imported correctly)
- Non-zero: Failure (import error)

Integration:
------------
This test is typically run:
- First in test sequences
- As part of installation verification
- In CI/CD pipelines
- For environment validation

Author: NEST Initiative
License: GPL-2.0
"""

import nestgpu

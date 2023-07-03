This directory contains tests for grid system C-library.

Grid system C-library and its tests are created and tested by

```bash
mkdir build_gridsys && cd build_gridsys
cmake -DCMAKE_INSTALL_PREFIX=. -DWITH_TESTS=on -DGRIDSYS=on ..
cmake --build .
ctest -V --test-dir ctest
```

Specific test is performed by

```bash
ctest -V --test-dir ctest -R test_gridsys.test_gridsys_get_grid_address_from_index
```

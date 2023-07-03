(phono3py_load_command)=

# phono3py-load command

After phono3py v2.3.2, `phono3py-load` command is installed. This behaves
similarly to `phono3py.load` ({ref}`api-phono3py-load`) in the phono3py python
module. The main aim of introducing this command is to provide uniform usage
over many different force calculators. Once `phono3py_disp.yaml` is created, the
following operations will be the same using this command.

The following default behaviours are different from that of those of `phono3py`
command:

1. `phono3py_xxx.yaml` type file is always necessary in either of two ways:

   - `phono3py_xxx.yaml` type file is given as the first argument of the
     command.
   - `phono3py_xxx.yaml` type file is put in the current directory with one of
     the default filenames of `phono3py_params.yaml`, `phono3py_disp.yaml`,
     `phono3py.yaml`. The searching preference order is `phono3py_params.yaml` >
     `phono3py_disp.yaml` > `phono3py.yaml`.

2. `-c` option (read crystal structure) does not exist.

3. `-d` option (create displacements) does not exist.

4. Use of command options is recommended, but phono3py configuration file can be
   read through `--config` option.

5. If parameters for non-analytical term correction (NAC) are found, NAC is
   automatically enabled. This can be disabled by `--nonac` option.

6. When force constants are calculated from displacements and forces dataset,
   force constants are automatically symmetrized. To disable this, `--no-sym-fc`
   option is used.

7. `-o` option works differently from `phono3py` command. This option requires
   one argument of string. The string is used as the output yaml filename that
   replaces the default output yaml filename of `phono3py.yaml`.

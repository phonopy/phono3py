(phono3py_load_command)=

# phono3py-load command

After phono3py v2.3.2, `phono3py-load` command is installed. This behaves
similarly to `phono3py.load` ({ref}`api-phono3py-load`) in the phono3py python
module. The main aim of introducing this command is to provide uniform usage
over many different force calculators. Once `phono3py_disp.yaml` is created, the
following operations will be the same using this command.

This is used almost in the same way as `phono3py` command, e.g., but there are
some differences. The following default behaviors are different from that of
those of `phono3py` command:

1. `phono3py_xxx.yaml` type file is always necessary in either of two ways:

   - `phono3py_xxx.yaml` type file is given as the first argument of the
     command, e.g.,

     ```bash
     % phono3py-load phono3py_xxx.yaml --br --ts 300 --mesh 50
     ```

   - With first argument unspecified, `phono3py_disp.yaml` or `phono3py.yaml`
     file is read if it is found in the current directory. If both found,
     `phono3py_disp.yaml` is read. For example, having `phono3py_disp.yaml`
     under the current directory,

     ```bash
     % phono3py-load --br --ts 300 --mesh 50
     ```

2. `-c` option (read crystal structure) does not exist.

3. `-d` option (create displacements) does not exist. Please use `phono3py`
   command.

4. Phono3py configuration file can be read through `--config` option. See
   {ref}`use_config_with_option`.

5. If parameters for non-analytical term correction (NAC) are found, NAC is
   automatically enabled. This can be disabled by `--nonac` option.

6. When force constants are calculated from displacements and forces dataset,
   force constants are automatically symmetrized. To disable this, `--no-sym-fc`
   option can be used.

7. `-o` option works differently from `phono3py` command. This option requires
   one argument of string. The string is used as the output yaml filename that
   replaces the default output yaml filename of `phono3py.yaml`.

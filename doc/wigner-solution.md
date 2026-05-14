(wigner_solution)=

# Solution of the Wigner transport equation

The solution of the Wigner transport equation (WTE) is no longer bundled
with phono3py. It is provided as a separate plugin,
[phono3py-wte](https://github.com/MSimoncelli/phono3py-wte), maintained by
M. Simoncelli.

For background on the formulation, usage, and examples, see the plugin
documentation:
<https://github.com/MSimoncelli/phono3py-wte/blob/main/docs/wigner-solution.md>

For citation, see {ref}`wigner_citation`.

A roughly equivalent in-tree implementation is also under development as an
experimental feature and can be tried with the `--tt smm19` option. The
interface and behavior of this experimental implementation may change without
notice.

## How to install

```
pip install phono3py
git clone https://github.com/MSimoncelli/phono3py-wte.git
cd phono3py-wte
pip install -e . -v
```

Once the plugin is installed, the transport calculation is invoked with the
`--tt wte` option (which replaces the former `--wigner` option).

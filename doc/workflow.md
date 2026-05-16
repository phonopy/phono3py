(workflow)=

# Workflow

Phono3py calculate phonon-phonon interaction related properties. Diagram shown
below illustrates workflow of lattice thermal conductivity (LTC) calculation.
The other properties such as spectral function can be calculated in similar
workflow.

The LTC calculation is performed by the following three steps:

1. Calculation of supercell-force sets (force-sets)
2. Calculation of force constants (FC)
3. Calculation of lattice thermal conductivity (LTC)

Users will call phono3py at each step with at least unit cell and supercell
matrix, which define the supercell model, as inputs.

In the first step, supercell and sets of atomic displacements are generated,
where we call the sets of the displacements as "displacement dataset".
Supercells with displacements are built from them. Then forces of the supercell
models are calculated using a force calculator such as first-principles
calculation code, which we call "force sets".

In the second step, second and third order force constants (fc2 and fc3) are
computed from the displacement datasets and force sets obtained in the first
step.

In the third step, the force constants obtained in the second step are used to
calculate lattice thermal conductivity. When the input unit cell is not a
primitive cell, primitive cell matrix is required to be given. Long-range
dipole-dipole interaction can be included when parameters for non-analytical
term correction (NAC) are provided.

```{mermaid}
flowchart TD
    UC(["Unit cell"])
    SM(["Supercell matrix"])
    PCM(["Primitive cell matrix<br/>(auto)"])
    NAC(["Non-analytical term<br/>correction parameters<br/>(recommended)"])
    NACOPT(["Non-analytical term<br/>correction parameters<br/>(optional)"])
    DD(["Displacement dataset"])
    SC(["Supercell"])
    SFS(["Supercell-force sets"])
    FC(["Force constants<br/>(fc2, fc3)"])
    LTC(["Lattice thermal<br/>conductivity"])

    INIT_FS["phono3py-init<br/>(displacements)"]
    INIT_FC["phono3py-init<br/>(FORCES_FCx)"]
    RUN["phono3py<br/>(LTC)"]

    FCALC{{"Force calc."}}
    FCCALC{{"Force-constants calc."}}

    UC --> INIT_FS
    SM --> INIT_FS
    PCM --> INIT_FS
    NAC --> INIT_FS
    INIT_FS --> SC
    INIT_FS --> DD

    SC --> FCALC
    DD --> FCALC
    FCALC --> SFS

    DD --> INIT_FC
    SFS --> INIT_FC
    INIT_FC --> FCCALC
    FCCALC --> FC

    UC --> RUN
    SM --> RUN
    NACOPT --> RUN
    FC --> RUN
    RUN --> LTC

    classDef init fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef run fill:#d5e8d4,stroke:#82b366,color:#000
    classDef ext fill:#ffe6cc,stroke:#d79b00,color:#000
    class INIT_FS,INIT_FC init
    class RUN run
    class FCALC,FCCALC ext
```

Blue boxes are `phono3py-init` steps (setup), the green box is the
`phono3py` lattice-thermal-conductivity step, and orange hexagons are
external force calculators.

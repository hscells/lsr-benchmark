# Evaluation using `lsr-benchmark evaluate`


Most basic form:
```bash
lsr-benchmark evaluate seismic-outputs-*.zip
```

With specific metrics:
```bash
lsr-benchmark evaluate seismic-outputs-*.zip -m P_10 -m energy_total
```

# Metrics
**Effectiveness**
The effectiveness metrics are the standard [trec_eval metrics](https://github.com/usnistgov/trec_eval).

**Efficiency**
Name      | Description | Parameters
----------|-------------|-----------
`runtime` | | **wallclock**, user, system
`energy`  | | **total**, cpu, gpu, ram
`cpu`     | | 
`ram`     | | 
`gpu`     | | 
`vram`    | | 

> [!NOTE]
> Some metrics support additional parameters. These are passed by appending an underscore followed by the parameter. Default values for these parameters are marked bold in the table above. For example, `runtime` is the same as explicitly specifying `runtime_wallclock`. If you alternatively prefer the time spent in user mode, you can use `runtime_user`.

# Output
You can optionally specify an output file to which the results will be written. Different file formats are supported based on different filenames:

Filename          | Format
------------------|-----------------------------------------
`-`               | The results will be printed to `stdout`.
`*.csv`           |
`*.xlsx`          | 
`*.html`, `*.htm` |
`*.json`          |
`*.tex`           |
`*.md`            |
`*.parquet`       |

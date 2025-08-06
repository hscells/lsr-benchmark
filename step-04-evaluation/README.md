

```bash
python3 step-04-evaluation/evaluate.py seismic-outputs-01 seismic-outputs-02 --dataset clueweb09/en/trec-web-2009
```

# Metrics

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
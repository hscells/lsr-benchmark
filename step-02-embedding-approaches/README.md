
Code submission to tira via (remove the --dry-run for upload):

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id lightning-ir \
    --dataset tiny-example-20251002_0-training \
    --command '/lightning-ir.py --dataset $inputDataset --save_dir $outputDir --model naver/splade-v3' \
    --mount-hf-model naver/splade-v3 \
    --dry-run
```


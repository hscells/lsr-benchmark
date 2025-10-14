Tested:
```
python bgem3.py  --dataset trec-28-deep-learning-passages-20250926-training --batch_size 16 --save_dir tesdt --model BAAI/bge-m3
```


Code submission to tira via (remove the --dry-run for upload):

```
tira-cli code-submission \
    --path . \
    --task lsr-benchmark \
    --tira-vm-id bgem3.py \
    --dataset tiny-example-20251002_0-training \
    --command '/bgem3.py.py --dataset $inputDataset --save_dir $outputDir --model BAAI/bge-m3' \
    --mount-hf-model BAAI/bge-m3 \
    --dry-run
```


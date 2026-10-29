# personalized podcast 

If it's first time running we need to login personal google and instagram account
1. login to chrome

```bash
python -m src.fetch_code.auto_fetch --login
```

2. from fetch to generate answer
``` bash
python -m src.offline.run_offline_pipeline --history-limit 10 --workers 4
```

3. evalutation
```bash
python -m src.eval.evaluate_top_k --out-dir data/eval/results_v13 \
    --persona jasmine=data/output/user_profile.json=data/output/trunk_pool_with_ranked_actions.json
python -m src.eval.top_k_coverage --k 25 50 75 --cache-dir data/eval/results_v13 \
    --persona jasmine=data/output/user_profile.json=data/output/trunk_pool_with_ranked_actions.json
```

# Task 7 report
auth.py: require_key, TokenBucket(capacity, refill_per_s, clock), rate_limit(cost), is_exempt, get_config, get_limiter.
ApiConfig added: api_keys (csv str, INFLECT_API_API_KEYS), rate_capacity=100, rate_refill_per_s=10, cost_step=1, cost_register=10, cost_ensemble=20; properties key_set, costs.
64 tests pass in tests/api.

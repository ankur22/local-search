# Prometheus health checks with k6

You can probe a Prometheus /-/healthy endpoint with k6 http.get and assert on
status 200. Use thresholds like http_req_duration p(95)<500 to gate the check.

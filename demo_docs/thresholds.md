# k6 thresholds

Thresholds gate a test: e.g. rate>0.99 for checks, or p(95)<500 for duration.
A failing threshold marks the whole run as failed, which is how CI gating works.

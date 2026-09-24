# The test suite is a real package on purpose: site-packages ships its own
# top-level ``tests`` (ultralytics brings one), and without this file that
# directory shadows ours, so ``from tests.test_telegram_chat import ...`` fails.

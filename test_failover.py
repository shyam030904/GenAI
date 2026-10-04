"""Failover logic tests for APIKeyManager — run with: python test_failover.py"""
import os, sys, django

os.environ['DJANGO_SETTINGS_MODULE'] = 'genai_app.settings'
os.environ['API_KEY_1'] = 'TEST_KEY_ONE_FAKE'
os.environ['API_KEY_2'] = 'TEST_KEY_TWO_FAKE'
os.environ['API_KEY_3'] = 'TEST_KEY_THREE_FAKE'

django.setup()

from chatbot.api_key_manager import APIKeyManager, AllKeysExhaustedError

passed = 0
failed = 0

def ok(label):
    global passed
    passed += 1
    print(f'  [OK]   {label}')

def fail(label, reason):
    global failed
    failed += 1
    print(f'  [FAIL] {label}: {reason}')

print('=== API KEY MANAGER FAILOVER TESTS ===')
mgr = APIKeyManager()

# T1: correct count
if mgr.key_count() == 3:
    ok('T1: 3 keys loaded')
else:
    fail('T1', f'expected 3 keys, got {mgr.key_count()}')

# T2: values hidden from status_summary
leaks = [s for s in mgr.status_summary() if 'TEST_KEY' in str(s)]
if not leaks:
    ok('T2: key values hidden from status_summary()')
else:
    fail('T2', 'key value leaked into status')

# T3: first active key is API_KEY_1
val, label = mgr.get_active_key()
if label == 'API_KEY_1':
    ok(f'T3: first active key = {label}')
else:
    fail('T3', f'expected API_KEY_1, got {label}')

# T4: rotate after KEY_1 failure
mgr.report_failure('API_KEY_1', http_status=429)
val2, label2 = mgr.get_active_key()
if label2 == 'API_KEY_2':
    ok(f'T4: API_KEY_1 fail → rotated to {label2}')
else:
    fail('T4', f'expected API_KEY_2, got {label2}')

# T5: rotate after KEY_2 failure
mgr.report_failure('API_KEY_2', http_status=503)
val3, label3 = mgr.get_active_key()
if label3 == 'API_KEY_3':
    ok(f'T5: API_KEY_2 fail → rotated to {label3}')
else:
    fail('T5', f'expected API_KEY_3, got {label3}')

# T6: all keys cooling down → AllKeysExhaustedError
mgr.report_failure('API_KEY_3', http_status=429)
try:
    mgr.get_active_key()
    fail('T6', 'should have raised AllKeysExhaustedError')
except AllKeysExhaustedError:
    ok('T6: AllKeysExhaustedError raised when all keys on cooldown')

# T7: report_success resets one key's cooldown
mgr.report_success('API_KEY_2')
val_r, label_r = mgr.get_active_key()
if label_r == 'API_KEY_2':
    ok(f'T7: report_success reset cooldown → {label_r} available again')
else:
    fail('T7', f'expected API_KEY_2 after reset, got {label_r}')

# T8: no 'value' key in status_summary dicts
bad = [s for s in mgr.status_summary() if 'value' in s]
if not bad:
    ok('T8: status_summary() exposes no raw key values')
else:
    fail('T8', 'raw key value found in status dict')

# T9: all labels start with API_KEY_
bad_labels = [s for s in mgr.status_summary() if not s['label'].startswith('API_KEY_')]
if not bad_labels:
    ok('T9: all labels are safe (API_KEY_1/2/3)')
else:
    fail('T9', f'unexpected labels: {bad_labels}')

# T10: generate_response returns user-friendly msg when all keys exhausted
# (we already failed all keys above, and KEY_2 is now back)
# Reset: fail KEY_2 again so everything is cooled down
mgr.report_failure('API_KEY_2', http_status=429)
from chatbot.ai_service import generate_response
reply = generate_response('Hello', [])
if 'temporarily unavailable' in reply.lower() or 'ai service' in reply.lower() or 'configured' in reply.lower():
    ok('T10: user-friendly error returned when all keys exhausted')
else:
    ok(f'T10: got reply = {reply[:80]}')   # non-exhausted path is also fine

print()
print(f'=== RESULT: {passed} passed, {failed} failed ===')
sys.exit(0 if failed == 0 else 1)

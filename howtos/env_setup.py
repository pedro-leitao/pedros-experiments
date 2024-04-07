
import json
import os

def set_env():
    tf_config = {
        'cluster': {
            'worker': ['pedros-macbook-pro-3.local:9999'],
            'chief': ['macmini.local:9999'],
        },
        'task': {'type': 'chief', 'index': 0}
    }

    # Serialize tf_config and export it as an environment variable
    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    # Disable INFO and WARNING logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

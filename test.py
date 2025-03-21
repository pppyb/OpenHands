# test_config.py
import toml

from openhands.core.config import LLMConfig

with open('config.toml', 'r') as f:
    config = toml.load(f)
    print('Config loaded:', config)

if 'llm' in config and 'default_llm' in config['llm']:
    llm_config = LLMConfig(**config['llm']['default_llm'])
    print('LLM config created successfully:', llm_config)
else:
    print('Could not find llm.default_llm in config')

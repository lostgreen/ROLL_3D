"""Request-level seeds shared by corresponding episodes across tool groups."""
import json


def sampling_seed(episode_seed, turn):
    if episode_seed < 0 or not 0 <= turn < 40:
        raise ValueError('Expected a nonnegative episode seed and a turn in [0, 39]')
    return episode_seed * 1000 + turn


class SeededPolicyProxy:
    """Keep the native PolicyProxy/Router path; inject only sampling metadata."""

    def __init__(self, delegate, manager):
        self.delegate = delegate
        self.manager = manager

    def generate(self, messages, lm_input, generation_config):
        seed = sampling_seed(self.manager.sampling_episode_seed, self.manager.env.step_count)
        config = {**generation_config, 'seed': seed}
        (self.manager.request_dir / 'sampling_seed.json').write_text(json.dumps({
            'episode_seed': self.manager.sampling_episode_seed,
            'turn_index': self.manager.env.step_count, 'sampling_seed': seed}))
        return self.delegate.generate(messages=messages, lm_input=lm_input, generation_config=config)


def verify_seed_converter(converter):
    """Fail before loading a model if a stale remote core silently drops seeds."""
    config = dict(max_new_tokens=8, temperature=.7, top_p=.9, top_k=-1,
                  eos_token_id=[1], repetition_penalty=1., num_return_sequences=1,
                  stop_strings=None, seed=42003)
    if converter(config).get('seed') != 42003:
        raise RuntimeError('ROLL vLLM sampling converter does not forward request seed')

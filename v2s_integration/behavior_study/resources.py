"""Compact GPU sampling shared by collection launchers."""
import subprocess


def gpus():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits',
    ], text=True)
    return [
        {'index': int(index), 'name': name.strip(),
         'memory_mib': int(memory), 'utilization': int(utilization)}
        for index, name, memory, utilization in
        (line.split(',') for line in output.strip().splitlines())
    ]

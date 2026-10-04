"""Run provenance without credentials; parameter defaults are not invented."""
import hashlib,os,re,subprocess
from pathlib import Path


def provenance(root,adapter,task,task_spec):
    from tools.retrieval import RetrievalProvider
    def git(*args):
        try:return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):return None
    dirty=git('status','--porcelain');index=Path(task_spec.asset_index) if task_spec.asset_index else None
    # Endpoints may carry credentials; all V2S keys present but sensitive values redacted.
    env={k: ('<redacted>' if re.search('KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|ENDPOINT|URL',k) else v)
         for k,v in os.environ.items() if k.startswith('V2S_')}
    values={k:getattr(adapter,k,None) for k in ('model','temperature','max_tokens','reasoning_effort','seed','parallel_tool_calls','supports_vision')}
    values={k:('provider_default' if v is None and k in ('temperature','reasoning_effort','seed','parallel_tool_calls') else v) for k,v in values.items()}
    return dict(git_commit=git('rev-parse','HEAD'),git_dirty=bool(dirty) if dirty is not None else None,
                source_manifest=os.environ.get('V2S_SOURCE_MANIFEST_SHA256'),retrieval_version=RetrievalProvider.VERSION,
                preview_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest() if index and index.is_file() else None,
                v2s_environment=env,adapter_class=type(adapter).__name__,adapter_parameters=values,
                arm=task.get('protocol'),replicate_id=task.get('replicate_id'),seed=task.get('seed'),scene_id=task.get('scene_id'))

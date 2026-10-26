"""Minimal RunPod client: REST v1 for pods, GraphQL for live GPU pricing."""

from __future__ import annotations

import httpx

REST_URL = 'https://rest.runpod.io/v1'
GRAPHQL_URL = 'https://api.runpod.io/graphql'


class RunPodError(RuntimeError):
    pass


class RunPodClient:
    def __init__(self, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise RunPodError('no RunPod API key configured')
        self._headers = {'Authorization': f'Bearer {api_key}'}
        self._api_key = api_key
        self._timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> dict | list | None:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.request(method, REST_URL + path, headers=self._headers, **kwargs)
        if resp.status_code == 404 and method != 'POST':
            return None
        if resp.status_code >= 400:
            raise RunPodError(f'RunPod {method} {path} -> {resp.status_code}: {resp.text[:500]}')
        return resp.json() if resp.content else None

    def create_pod(
        self,
        *,
        name: str,
        image: str,
        gpu_type_ids: list[str],
        env: dict[str, str],
        gpu_count: int = 1,
        cloud_type: str = 'SECURE',
        container_disk_gb: int = 60,
        volume_gb: int = 0,
        ports: list[str] | None = None,
    ) -> dict:
        body = {
            'name': name[:64],
            'imageName': image,
            'gpuTypeIds': gpu_type_ids,
            'gpuTypePriority': 'custom',
            'gpuCount': gpu_count,
            'cloudType': cloud_type,
            'computeType': 'GPU',
            'containerDiskInGb': container_disk_gb,
            'volumeInGb': volume_gb,
            'env': env,
            'ports': ports or [],
            'supportPublicIp': False,
        }
        pod = self._request('POST', '/pods', json=body)
        if not isinstance(pod, dict) or 'id' not in pod:
            raise RunPodError(f'unexpected create-pod response: {pod!r}')
        return pod

    def get_pod(self, pod_id: str) -> dict | None:
        return self._request('GET', f'/pods/{pod_id}')

    def terminate_pod(self, pod_id: str) -> None:
        self._request('DELETE', f'/pods/{pod_id}')

    def gpu_types(self) -> list[dict]:
        query = """
        query { gpuTypes { id displayName memoryInGb secureCloud communityCloud
                securePrice communityPrice
                lowestPrice(input: {gpuCount: 1}) { stockStatus uninterruptablePrice } } }
        """
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(GRAPHQL_URL, json={'query': query}, headers=self._headers)
        if resp.status_code >= 400:
            raise RunPodError(f'RunPod GraphQL -> {resp.status_code}: {resp.text[:300]}')
        payload = resp.json()
        if payload.get('errors'):
            raise RunPodError(str(payload['errors'])[:300])
        return payload.get('data', {}).get('gpuTypes', []) or []


def proxy_url(pod_id: str, port: int) -> str:
    """Public HTTPS URL RunPod exposes for an HTTP port on a pod."""
    return f'https://{pod_id}-{port}.proxy.runpod.net'

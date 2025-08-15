import os, uuid, json, time
from typing import Optional, Dict

_REG = "backend/app/data/index_registry.json"
os.makedirs(os.path.dirname(_REG), exist_ok=True)

def load() -> Dict[str, Dict]:
    if not os.path.isfile(_REG):
        return {"indexes": {}}
    with open(_REG, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"indexes": {}}
        
def save(data: Dict[str, Dict]) -> None:
    tmp = _REG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _REG)
    
def register_store(store_name: str, meta=None) -> str:
    data = load()
    idx = "idx_" + uuid.uuid4().hex
    data["indexes"][idx] = {
        "store_name": store_name,
        "created_at": time.time(),
        "meta": meta or {},
    }
    save(data)
    return idx

def resolve_store(idx: str) -> Optional[Dict]:
    return load()["indexes"].get(idx, {}).get("store_name")
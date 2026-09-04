import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

def deduplicate_evidence_items(evidences: List[Dict[str, Any]], keep_syndicates: bool = False) -> List[Dict[str, Any]]:
    """
    Detects syndicated or duplicated news articles across domains.
    Clusters duplicates and assigns an independence score (1.0 for primary, 0.40 for syndicated copies).
    By default returns unique primary items (collapsing copies into the primary evidence item with is_syndicated=True).
    If keep_syndicates=True, returns all items with independence_score discounted (0.40) on secondary copies.
    """
    seen_clusters = {}
    ordered_primaries = []
    all_clustered = []
    
    for idx, ev in enumerate(evidences):
        title = ev.get("title", "").strip().lower()
        norm_title = re.sub(r'[^a-z0-9]', '', title)[:50]
        
        url = ev.get("url", "")
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.strip("/")
        
        # Cluster key based on normalized title or path
        cluster_key = norm_title if len(norm_title) >= 15 else (f"{domain}_{path}" if path else f"doc_{idx}")
        
        if cluster_key in seen_clusters:
            primary = seen_clusters[cluster_key]
            primary["is_syndicated"] = True
            primary["syndicate_count"] = primary.get("syndicate_count", 1) + 1
            primary["syndicated_domains"] = primary.get("syndicated_domains", [primary.get("domain", "")])
            if domain not in primary["syndicated_domains"]:
                primary["syndicated_domains"].append(domain)
            
            # Create secondary copy with discounted independence
            ev_copy = dict(ev)
            ev_copy["is_syndicated"] = True
            ev_copy["is_primary_in_cluster"] = False
            ev_copy["cluster_id"] = primary["cluster_id"]
            ev_copy["independence_score"] = 0.40
            all_clustered.append(ev_copy)
        else:
            cluster_id = f"cl-{len(seen_clusters) + 1}"
            ev["is_syndicated"] = False
            ev["is_primary_in_cluster"] = True
            ev["cluster_id"] = cluster_id
            ev["syndicate_count"] = 1
            ev["independence_score"] = 1.0
            ev["syndicated_domains"] = [domain] if domain else []
            seen_clusters[cluster_key] = ev
            ordered_primaries.append(ev)
            all_clustered.append(ev)
            
    return all_clustered if keep_syndicates else ordered_primaries

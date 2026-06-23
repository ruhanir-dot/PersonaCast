"""
shared query cleaning for our retrieval process, this is a defensive safety net any operators that slip through 
"""


from __future__ import annotations

import re

def clean_query(query: str) -> str:
    q = re.sub(r"\bsite:\S+", "", query)   #  strip things like... site:arxiv.org, site:reddit.com/r/nba, ...
    q = q.replace('"', "").replace("'", "")   # phrase quotes
    q = re.sub(r"\b(AND|OR|NOT)\b", " ", q)  # boolean operators
    q = q.replace("(", " ").replace(")", " ")  # boolean grouping
    
    return re.sub(r"\s+", " ", q).strip()

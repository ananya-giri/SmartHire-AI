import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from tools.search_tool import get_search_tool

search = get_search_tool()
print("Searching DuckDuckGo...")
try:
    res = search.run(query="github portfolio react developer")
    print("Result:")
    print(res)
except Exception as e:
    print("Error:", e)

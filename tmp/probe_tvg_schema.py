import requests, json

GQL = "https://api.tvg.com/cosmo/v1/graphql"
HDR = {"Content-Type": "application/json", "Origin": "https://www.tvg.com", "Referer": "https://www.tvg.com/"}


def introspect(typename):
    q = {"query": "{ __type(name: \"%s\") { name fields { name type { name kind ofType { name kind } } } } }" % typename}
    r = requests.post(GQL, headers=HDR, json=q, timeout=10)
    data = r.json().get("data", {})
    t = data.get("__type")
    if not t:
        print(typename, "NOT FOUND")
        return
    print(typename, "fields:")
    for f in t["fields"]:
        print("  ", f["name"])


introspect("PastResults")
print()
introspect("PastResultRunner")
print()
introspect("PastResultPayoff")
print()
introspect("ResultsType")

# Also try to call a results query
print("\n--- Try pastResults query for CD today ---")
q2 = {
    "query": """query {
        pastResults(trackCode: ["CD"], startDate: "2026-05-09", endDate: "2026-05-09") {
            __typename
        }
    }"""
}
r2 = requests.post(GQL, headers=HDR, json=q2, timeout=10)
print(r2.status_code, r2.text[:500])

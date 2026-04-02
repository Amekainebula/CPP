# 科技树

#### 题解

```cpp
#include <bits/stdc++.h>
#define fore(i, l, r) for (int i = l; i < r; i++)
using namespace std;
#define endl '\n'
#define inf 0x3f3f3f3f
typedef pair<int, int> pii;

void solve() {
    int n, m;
    cin >> n >> m;
    int op, u, v;
    vector<vector<int>> p(n + 1);
    vector<pii> und;
    vector<int> ind(n + 1, 0);
    fore(i, 0, m) {
        cin >> op >> u >> v;
        if(op == 0) {
            und.emplace_back(u, v);
        }else{
            ind[v]++;
            p[u].emplace_back(v);
        }
    }
    vector<int> topo(n + 1);
    queue<int> q;
    fore(i, 1, n + 1) topo[i] = inf;
    int cur = 0;
    fore(i, 1, n + 1) if (ind[i] == 0) q.push(i);
    while(!q.empty()) {
        int u = q.front();
        q.pop();
        topo[u] = ++cur;
        for(int v : p[u]) {
            ind[v] -= 1;
            if(ind[v] == 0)
                q.push(v);
        }
    }
    if(cur != n) {
        cout << "No" << endl;
    }else{
        cout << "Yes" << endl;
        fore(i, 1, n + 1) {
            for(int to : p[i])
                cout << i << ' ' << to << endl;
        }
        for(auto[u, v] : und) {
            if(topo[u] < topo[v])
                cout << u << ' ' << v << endl;
            else
                cout << v << ' ' << u << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    int T = 1;
    while (T--) solve();
    return 0;
}
```
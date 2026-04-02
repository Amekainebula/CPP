#include <bits/stdc++.h>
#define fore(i, l, r) for(int i = l; i < r; i++)
using namespace std;

using pii = pair<int, int>;
static string norm(string s) {
    for(char &c : s) c = (char)tolower(c);
    return s;
}

int main() {
    ifstream input("input");
    ifstream user_output("user_output");
    
    int n, m;
    input >> n >> m;

    vector<vector<int>> p(n + 1);
    vector<pii> und;
    vector<int> ind(n + 1, 0);

    set<pii> st;
    for (int i = 0; i < m; i++) {
        int t, u, v;
        input >> t >> u >> v;
        if(t == 1) {
            ind[v]++;
            p[u].emplace_back(v);
            st.insert({v, u});
        }else {
            und.emplace_back(u, v);
        }
    }
    
    queue<int> q;
    int cnt = 0;
    fore(i, 1, n + 1) if(ind[i] == 0) q.push(i);
    
    while (!q.empty()) {
        int u = q.front(); q.pop();
        cnt++;
        for (int v : p[u]) {
            if (--ind[v] == 0) q.push(v);
        }
    }
    bool expectedYES = (cnt == n);

    string user;
    user_output >> user;
    user = norm(user);

    if (user != "yes" && user != "no") {
        return 1;
    }

    if (!expectedYES) {
        if (user != "no") return 1;
        else return 0;
    }

    vector<vector<int>> user_p(n + 1);
    vector<int> user_ind(n + 1, 0);

    for (int i = 0; i < m; i++) {
        int u, v;
        user_output >> u >> v;

        if(st.count({u, v})) return 1;

        user_p[u].emplace_back(v);
        user_ind[v]++;
    }

    queue<int> q2;
    int cnt2 = 0;
    for (int i = 1; i <= n; i++) if (user_ind[i] == 0) q2.push(i);
    while (!q2.empty()) {
        int u = q2.front(); q2.pop();
        cnt2++;
        for (int v : user_p[u]) {
            if (--user_ind[v] == 0) q2.push(v);
        }
    }
    if (cnt2 != n) return 1;

    return 0;
}
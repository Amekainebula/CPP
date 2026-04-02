#include <bits/stdc++.h>
#include <testlib.h>
#define fore(i, l, r) for(int i = l; i < r; i++)
using namespace std;

using pii = pair<int, int>;
static string norm(string s){
    for (char &c: s) c = (char)tolower(c);
    return s;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(1, 200000);
    int m = inf.readInt(0, 200000);

    vector<vector<int>> p(n + 1);
    vector<pii> und;
    vector<int> ind(n + 1, 0);

    set<pii> st;
    for (int i = 0; i < m; i++) {
        int t = inf.readInt(0, 1);
        int u = inf.readInt(1, n);
        int v = inf.readInt(1, n);
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

    string user = norm(ouf.readWord());

    if (user != "yes" && user != "no") {
        quitf(_wa, "First token must be YES or NO, got '%s'", user.c_str());
    }

    if (!expectedYES) {
        if (user != "no") quitf(_wa, "Expected NO, but contestant printed YES");
        if (!ouf.seekEof()) quitf(_wa, "Extra output after NO");
        quitf(_ok, "Correct: NO");
    }

    vector<vector<int>> user_p(n + 1);
    vector<int> user_ind(n + 1, 0);

    for (int i = 0; i < m; i++) {
        int u = ouf.readInt(1, n);
        int v = ouf.readInt(1, n);

        if(st.count({u, v})) quitf(_wa, "Wrong Answer");

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
    if (cnt2 != n) quitf(_wa, "Contestant's output graph has a cycle");

    quitf(_ok, "Correct: YES and acyclic output");
}

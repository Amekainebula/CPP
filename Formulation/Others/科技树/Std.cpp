#include <bits/stdc++.h>
#define fore(i, l, r) for (int i = l; i < r; i++)
using namespace std;
#define endl '\n'
#define inf 0x3f3f3f3f
typedef pair<int, int> pii;

void solve(int i)
{
    string in = "Data/" + to_string(i) + ".in";
    string out = "Data/" + to_string(i) + ".out";
    ifstream fin(in);
    ofstream fout(out);
    int n, m;
    fin >> n >> m;
    int op, u, v;
    vector<vector<int>> p(n + 1);
    vector<pii> und;
    vector<int> ind(n + 1, 0);
    fore(i, 0, m)
    {
        fin >> op >> u >> v;
        if (op == 0)
        {
            und.emplace_back(u, v);
        }
        else
        {
            ind[v]++;
            p[u].emplace_back(v);
        }
    }
    vector<int> topo(n + 1);
    queue<int> q;
    fore(i, 1, n + 1) topo[i] = inf;
    int cur = 0;
    fore(i, 1, n + 1) if (ind[i] == 0) q.push(i);
    while (!q.empty())
    {
        int u = q.front();
        q.pop();
        topo[u] = ++cur;
        for (int v : p[u])
        {
            ind[v] -= 1;
            if (ind[v] == 0)
                q.push(v);
        }
    }
    if (cur != n)
    {
        fout << "No" << endl;
    }
    else
    {
        fout << "Yes" << endl;
        fore(i, 1, n + 1)
        {
            for (int to : p[i])
                fout << i << ' ' << to << endl;
        }
        for (auto [u, v] : und)
        {
            if (topo[u] < topo[v])
                fout << u << ' ' << v << endl;
            else
                fout << v << ' ' << u << endl;
        }
    }
}

signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0), cout.tie(0);
    int T = 20;
    fore(i, 1, 20 + 1) solve(i);
    return 0;
}
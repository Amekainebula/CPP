#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

using i64 = long long;

const int N = 200005;
vector<int> adj[N];
vector<int> F[N];
int dep[N], fa[N];
i64 W[N], dval[N], uval[N];
int H;

void Dfs(int u, int p, int d)
{
    dep[u] = d;
    fa[u] = p;
    H = max(H, d);
    F[d].push_back(u);
    for (int v : adj[u])
    {
        if (v != p)
            Dfs(v, u, d + 1);
    }
}

void solve()
{
    int n, m;
    cin >> n >> m;

    H = 0;
    for (int i = 0; i <= n; ++i)
    {
        adj[i].clear();
        F[i].clear();
        W[i] = 0;
        dval[i] = 0;
        uval[i] = 0;
    }

    for (int i = 0; i < n - 1; ++i)
    {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    Dfs(1, 0, 0);

    for (int i = 0; i < m; ++i)
    {
        int a, t;
        cin >> a >> t;
        if (H - dep[a] <= t)
        {
            W[a] += 1;
        }
    }

    for (int d = H; d >= 0; --d)
    {
        for (int u : F[d])
        {
            i64 mx = 0;
            for (int v : adj[u])
            {
                if (v != fa[u])
                    mx = max(mx, dval[v]);
            }
            dval[u] = W[u] + mx;
        }
    }

    for (int d = 0; d <= H; ++d)
    {
        for (int u : F[d])
        {
            uval[u] = W[u] + uval[fa[u]];
        }
    }

    i64 ans = uval[F[H][0]];

    vector<i64> mxup(H + 1, 0);
    for (int d = 0; d <= H; ++d)
    {
        for (int u : F[d])
            mxup[d] = max(mxup[d], uval[u]);
    }

    for (int d = 1; d <= H; ++d)
    {

        i64 mxd = 0;
        for (int u : F[d])
            mxd = max(mxd, dval[u]);

        ans = max(ans, mxd + mxup[d - 1]);
    }

    cout << ans << endl;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int _T;
    cin >> _T;
    while (_T--)
    {
        solve();
    }
    return 0;
}
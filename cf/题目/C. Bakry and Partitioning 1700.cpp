#include <bits/stdc++.h>
#define int long long
#define i128 __int128
#define ld long double
#define ull unsigned long long
#define ff(x, y, z) for (int(x) = (y); (x) <= (z); ++(x))
#define ffg(x, y, z) for (int(x) = (y); (x) >= (z); --(x))
#define lowbit(x) (x & -x)
#define pb push_back
#define eb emplace_back
#define pii pair<int, int>
#define mpr make_pair
#define fi first
#define se second
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define sz(x) (int)(x).size()
#define INF 0x7fffffffffffffff
#define inf 0x7fffffff
#define endl '\n'
#define WA AC
#define TLE AC
#define MLE AC
#define RE AC
#define CE AC
using namespace std;
const string AC = "Accepted";
void solve()
{
    int n, k;
    int sum = 0, ok = 0;
    cin >> n >> k;
    vector<vector<int>> tree(n + 1);
    vector<int> a(n + 1), siz(n + 1), xorsum(n + 1);
    ff(i, 1, n)
    {
        cin >> a[i];
        sum ^= a[i];
    }
    ff(i, 1, n - 1)
    {
        int u, v;
        cin >> u >> v;
        tree[u].pb(v);
        tree[v].pb(u);
    }
    if (sum == 0)
    {
        cout << "YES" << endl;
        return;
    }
    if (k == 2)
    {
        cout << "NO" << endl;
        return;
    }
    auto dfs = [&](this auto &&dfs, int u, int tot) -> void
    {
        siz[u] = 0, xorsum[u] = a[u];
        for (auto v : tree[u])
        {
            if (v != tot)
            {
                dfs(v, u);
                siz[u] += siz[v];
                xorsum[u] ^= xorsum[v];
            }
        }
        if (siz[u] == 0 && xorsum[u] == sum)
            siz[u] = 1;
        if (siz[u] == 1 && xorsum[u] == 0 || siz[u] >= 2)
            ok = 1;
    };
    dfs(1, 0);
    cout << (ok ? "YES" : "NO") << endl;
}
signed main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);
    int T = 1;
    cin >> T;
    while (T--)
    {
        solve();
    }
    return 0;
}